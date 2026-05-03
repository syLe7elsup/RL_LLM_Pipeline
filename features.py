"""Feature engineering for POLAR trajectories.

Raw POLAR trajectory layout (per row, total 11 dims):
    [0:2]   s_1   stage-1 patient indicators
    [2]     a_1   stage-1 treatment (0/1)
    [3:5]   s_2   stage-2 patient indicators
    [5]     a_2   stage-2 treatment
    [6:8]   s_3   stage-3 patient indicators
    [8]     a_3   stage-3 treatment
    [9:11]  s_4   final patient indicators (this is the future to predict)

The classification setup we use is:
    INPUT  = the *observed* trajectory through stage 3 (s_1..s_3 and a_1..a_3)
    LABEL  = (reward(s_3, s_4) > median), where s_4 is the *unobserved future*

Crucially, s_4 must NOT appear in the input. The reward is a closed-form
function of s_3 and s_4, so any feature that includes s_4 leaks the label
and reduces the task to inverting a known formula (we saw 99% accuracy and
SAE collapse from this). Dropping s_4 keeps the problem an honest
prediction task with real stochasticity from the s_3 -> s_4 transition.

Each feature is given a human-readable name (returned by ``feature_names``).
The names are reused downstream so LLM1's evidence prompts can reference
them by name (e.g., "s2_ind2 high, delta2_ind1 negative").
"""

from __future__ import annotations

import numpy as np


def _split(traj: np.ndarray):
    s1 = traj[:, 0:2]
    a1 = traj[:, 2:3]
    s2 = traj[:, 3:5]
    a2 = traj[:, 5:6]
    s3 = traj[:, 6:8]
    a3 = traj[:, 8:9]
    s4 = traj[:, 9:11]
    return s1, a1, s2, a2, s3, a3, s4


def feature_names() -> list[str]:
    """Names of input features (s_4 deliberately excluded — it's the future)."""
    return [
        # raw indicators through stage 3 (6)
        "s1_ind1", "s1_ind2",
        "s2_ind1", "s2_ind2",
        "s3_ind1", "s3_ind2",
        # raw treatments (3)
        "a1", "a2", "a3",
        # stage deltas through stage 3 (4)
        "delta1_ind1", "delta1_ind2",  # s2 - s1
        "delta2_ind1", "delta2_ind2",  # s3 - s2
        # treatment-pattern features (5)
        "treatment_count",   # a1 + a2 + a3   (0..3)
        "all_treated",       # 1 iff a1=a2=a3=1
        "never_treated",     # 1 iff a1=a2=a3=0
        "switch_count",      # |a1-a2| + |a2-a3|
        "early_treated",     # 1 iff a1=1
        # state-trajectory summaries over s_1..s_3 (6)
        "mean_ind1", "mean_ind2",
        "max_ind1",  "max_ind2",
        "min_ind1",  "min_ind2",
        # interactions (4)
        "s1_x_s3_ind1", "s1_x_s3_ind2",  # initial-vs-current product
        "ind1_range",                     # max_ind1 - min_ind1 over s_1..s_3
        "ind2_range",                     # max_ind2 - min_ind2 over s_1..s_3
    ]


def build_features(traj: np.ndarray) -> np.ndarray:
    """Expand (n, 11) raw trajectories into (n, F) named feature vectors.

    Output dimension is 28. s_4 is intentionally excluded — see module docstring.
    The order matches ``feature_names()``.
    """
    if traj.ndim != 2 or traj.shape[1] != 11:
        raise ValueError(f"Expected (n, 11) trajectories, got {traj.shape}")

    s1, a1, s2, a2, s3, a3, _s4 = _split(traj)

    treatment_count = (a1 + a2 + a3).reshape(-1, 1)
    all_treated = ((a1 == 1) & (a2 == 1) & (a3 == 1)).astype(np.float32)
    never_treated = ((a1 == 0) & (a2 == 0) & (a3 == 0)).astype(np.float32)
    switch_count = (np.abs(a1 - a2) + np.abs(a2 - a3)).astype(np.float32)
    early_treated = (a1 == 1).astype(np.float32)

    states = np.stack([s1, s2, s3], axis=1)  # (n, 3, 2)
    mean_state = states.mean(axis=1)
    max_state = states.max(axis=1)
    min_state = states.min(axis=1)

    s1_x_s3 = s1 * s3
    range_state = max_state - min_state

    feats = np.concatenate(
        [
            s1, s2, s3,                                   # 6
            a1, a2, a3,                                   # 3
            s2 - s1, s3 - s2,                             # 4
            treatment_count, all_treated, never_treated,  # 3
            switch_count, early_treated,                  # 2
            mean_state, max_state, min_state,             # 6
            s1_x_s3, range_state,                         # 4
        ],
        axis=1,
    ).astype(np.float32)

    expected = len(feature_names())
    if feats.shape[1] != expected:
        raise RuntimeError(
            f"Feature count mismatch: {feats.shape[1]} vs feature_names() {expected}"
        )
    return feats


def compute_reward(traj: np.ndarray) -> np.ndarray:
    """POLAR's stage-3 reward, evaluated on observed trajectories.

    Mirrors ``polar_data.r_bar`` for k = k_max = 3:
        reward = cos(-pi*s3_1) + 2*cos(pi*s3_2) + s4_1 + 2*s4_2
        normalized as (reward - 1.37) * 3.8
    """
    _, _, _, _, s3, _, s4 = _split(traj)
    s3_1, s3_2 = s3[:, 0], s3[:, 1]
    s4_1, s4_2 = s4[:, 0], s4[:, 1]
    raw = np.cos(-np.pi * s3_1) + 2 * np.cos(np.pi * s3_2) + s4_1 + 2 * s4_2
    return (raw - 1.37) * 3.8


def reward_to_outcome_label(reward: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    """Convert continuous rewards to a balanced binary high/low outcome label.

    If ``threshold`` is None, uses the median (balanced classes).
    Returns (labels, threshold_used).
    """
    if threshold is None:
        threshold = float(np.median(reward))
    labels = (reward > threshold).astype(np.int64)
    return labels, threshold

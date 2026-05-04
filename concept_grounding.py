"""Stage 10: ground SAE features against the ARF simulator's true latent
concepts.

Why this matters: the ARF simulator (``arf_data.simulate_toy_arf_data``)
generates each (patient, timestep)'s observed variables from 6 latent
concepts:

    oxygenation_failure        (c1)
    ventilatory_failure        (c2)
    metabolic_stress           (c3)
    hemodynamic_instability    (c4)
    inflammation_severity      (c5)
    recovery_trend             (c6)

The DQN never sees these latents — it only sees the 27 observed variables.
The SAE then sparsifies the DQN's hidden representation. If the SAE has
recovered useful structure, its sparse latents should correlate with the
six ground-truth concepts (or sums / differences thereof).

This module provides:
    - ``load_ground_truth_concepts``: line up concept_df rows with our
      train/val/test SAE latents by (patient_id, time_step).
    - ``compute_grounding_matrix``: |z_alive| × 6 matrix of |Pearson r|.
    - ``best_match_per_concept``: for each ground-truth concept, the SAE
      feature that explains it best.
    - ``best_match_per_feature``: for each SAE feature, its most-aligned
      ground-truth concept.

This is "free" — no LLM calls, just numpy correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GROUND_TRUTH_COLS = [
    "oxygenation_failure",
    "ventilatory_failure",
    "metabolic_stress",
    "hemodynamic_instability",
    "inflammation_severity",
    "recovery_trend",
]


def load_ground_truth_concepts(concept_csv: str | Path) -> pd.DataFrame:
    """Read ``toy_arf_ground_truth_concepts.csv`` produced by LLM_RL stage 1.

    Columns: ``patient_id``, ``time_step``, plus the six concept columns.
    """
    df = pd.read_csv(concept_csv)
    missing = set(GROUND_TRUTH_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"concept CSV missing columns: {missing}")
    return df


def align_concepts_to_latents(
    concept_df: pd.DataFrame,
    patient_ids,
    time_steps,
) -> np.ndarray:
    """Look up the 6 ground-truth concept values for each (pid, ts) row of
    SAE latents. Returns ``(N, 6)`` float matrix in the order of
    ``patient_ids`` / ``time_steps``.
    """
    if len(patient_ids) != len(time_steps):
        raise ValueError("patient_ids and time_steps must match length")

    keyed = concept_df.set_index(["patient_id", "time_step"])
    rows = []
    missing = 0
    for pid, ts in zip(patient_ids, time_steps):
        try:
            r = keyed.loc[(int(pid), int(ts))]
            rows.append(r[GROUND_TRUTH_COLS].values.astype(float))
        except KeyError:
            rows.append(np.full(len(GROUND_TRUTH_COLS), np.nan))
            missing += 1
    if missing:
        print(f"[grounding] warning: {missing} (pid, ts) pairs not found in concept_df")
    return np.asarray(rows, dtype=np.float64)


def compute_grounding_matrix(
    Z: np.ndarray,
    C: np.ndarray,
    feature_indices: list[int] | None = None,
    *,
    use_abs: bool = True,
) -> np.ndarray:
    """Pearson correlation between every SAE feature and every concept.

    Args:
        Z:                (N, latent_dim) SAE activations.
        C:                (N, K) ground-truth concept values.
        feature_indices:  subset of features to evaluate (default: all alive
            features, defined as activation > 0 on at least one row).
        use_abs:          report |r|. Direction is recoverable from the raw
            matrix returned by ``compute_signed_grounding_matrix``.

    Returns:
        ``(F, K)`` matrix. Row order = ``feature_indices``.
    """
    if feature_indices is None:
        feature_indices = [i for i in range(Z.shape[1]) if (Z[:, i] > 0).sum() > 0]

    out = np.zeros((len(feature_indices), C.shape[1]), dtype=np.float64)
    valid = ~np.isnan(C).any(axis=1)
    Zv = Z[valid]
    Cv = C[valid]

    for j_out, fid in enumerate(feature_indices):
        z = Zv[:, fid]
        if z.std() < 1e-12:
            continue
        for k in range(Cv.shape[1]):
            c = Cv[:, k]
            if c.std() < 1e-12:
                continue
            r = np.corrcoef(z, c)[0, 1]
            if np.isnan(r):
                continue
            out[j_out, k] = abs(r) if use_abs else r
    return out


def compute_signed_grounding_matrix(Z, C, feature_indices=None):
    """Same as ``compute_grounding_matrix`` but signed (no abs)."""
    return compute_grounding_matrix(Z, C, feature_indices, use_abs=False)


# ---------------------------------------------------------------------------
# Convenience reporting helpers


@dataclass
class FeatureMatch:
    feature_idx: int
    concept: str
    correlation: float
    rank_within_feature: int  # 0 = top match for this feature


def best_match_per_feature(
    grounding: np.ndarray,
    feature_indices: list[int],
) -> dict[int, FeatureMatch]:
    """For each SAE feature, return its strongest ground-truth match."""
    out = {}
    for j_out, fid in enumerate(feature_indices):
        row = grounding[j_out]
        order = np.argsort(-row)  # desc
        k_best = int(order[0])
        out[fid] = FeatureMatch(
            feature_idx=fid,
            concept=GROUND_TRUTH_COLS[k_best],
            correlation=float(row[k_best]),
            rank_within_feature=0,
        )
    return out


def best_match_per_concept(
    grounding: np.ndarray,
    feature_indices: list[int],
) -> dict[str, FeatureMatch]:
    """For each ground-truth concept, return the SAE feature most aligned."""
    out = {}
    for k, name in enumerate(GROUND_TRUTH_COLS):
        col = grounding[:, k]
        if col.size == 0:
            continue
        j_best = int(np.argmax(col))
        out[name] = FeatureMatch(
            feature_idx=feature_indices[j_best],
            concept=name,
            correlation=float(col[j_best]),
            rank_within_feature=-1,  # not meaningful here
        )
    return out


def report_grounding(
    grounding: np.ndarray,
    feature_indices: list[int],
    *,
    concept_labels: dict[int, str] | None = None,
    top_n_per_concept: int = 3,
) -> str:
    """Build a tidy text report of the grounding matrix."""
    lines = []
    lines.append("=" * 64)
    lines.append("Grounding: SAE features ↔ ground-truth latent concepts")
    lines.append("=" * 64)
    lines.append(f"{len(feature_indices)} alive SAE features × {grounding.shape[1]} concepts")

    lines.append("\nBest match per ground-truth concept (top {} SAE features):".format(top_n_per_concept))
    for k, name in enumerate(GROUND_TRUTH_COLS):
        col = grounding[:, k]
        order = np.argsort(-col)[:top_n_per_concept]
        bits = []
        for j_out in order:
            fid = feature_indices[j_out]
            tag = ""
            if concept_labels and fid in concept_labels:
                tag = f' "{concept_labels[fid]}"'
            bits.append(f"i{fid}({col[j_out]:.2f}){tag}")
        lines.append(f"  {name:<28s}  →  " + ", ".join(bits))

    lines.append("\nPer-feature best match (sorted by strength):")
    feat_match = best_match_per_feature(grounding, feature_indices)
    sorted_feats = sorted(feat_match.values(), key=lambda m: -m.correlation)
    for m in sorted_feats:
        tag = ""
        if concept_labels and m.feature_idx in concept_labels:
            tag = f'  LLM1: "{concept_labels[m.feature_idx]}"'
        lines.append(
            f"  i{m.feature_idx:>3d}  →  {m.concept:<28s}  |r|={m.correlation:.3f}{tag}"
        )

    lines.append("\nSummary:")
    lines.append(
        f"  features with |r| ≥ 0.30 to some concept: "
        f"{sum(1 for m in feat_match.values() if m.correlation >= 0.30)}/{len(feat_match)}"
    )
    lines.append(
        f"  features with |r| ≥ 0.50 to some concept: "
        f"{sum(1 for m in feat_match.values() if m.correlation >= 0.50)}/{len(feat_match)}"
    )
    lines.append(
        f"  mean best |r|: {np.mean([m.correlation for m in feat_match.values()]):.3f}"
    )
    return "\n".join(lines)

"""Per-SAE-feature directional polarity wrt the binary outcome label.

Without polarity, LLM2 has no way to know whether activating feature i pushes
the model toward HIGH or LOW outcome — it tends to invent directional
reasoning ("strong positive trend") that contradicts the blackbox prediction
(observed: KL = 18 in one run).

Polarity is computed once on training data using point-biserial correlation
between the binary activation (z_i > 0) and the binary label y. Returned as
a small typed object so the LLM2 prompt can render it consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeaturePolarity:
    feature_idx: int
    correlation: float              # point-biserial in [-1, 1]
    direction: str                  # "HIGH" / "LOW" / "NEUTRAL"
    strength: str                   # "weak" / "moderate" / "strong"
    p_high_when_active: float       # P(y=high | z_i > 0)
    p_high_when_inactive: float     # P(y=high | z_i == 0)


def _label_strength(corr: float) -> str:
    a = abs(corr)
    if a < 0.10:
        return "weak"
    if a < 0.30:
        return "moderate"
    return "strong"


def compute_polarities(
    Z: np.ndarray,
    y: np.ndarray,
    feature_indices: list[int] | None = None,
    *,
    neutral_threshold: float = 0.05,
) -> dict[int, FeaturePolarity]:
    """For each feature index, compute its directional polarity wrt y.

    Args:
        Z: (n, latent_dim) sparse SAE latent matrix from ``encode_all``.
        y: (n,) binary labels (0 = LOW outcome, 1 = HIGH outcome).
        feature_indices: subset of features to score; defaults to all latents
            with at least one active row.
        neutral_threshold: |corr| below this is reported as NEUTRAL.

    Returns:
        ``{feature_idx: FeaturePolarity}`` for each feature scored.
    """
    if Z.shape[0] != y.shape[0]:
        raise ValueError(f"Z rows {Z.shape[0]} != y rows {y.shape[0]}")
    n, latent_dim = Z.shape
    y = y.astype(np.float64)
    p_high = float(y.mean())

    if feature_indices is None:
        feature_indices = [i for i in range(latent_dim) if (Z[:, i] > 0).sum() > 0]

    out: dict[int, FeaturePolarity] = {}
    for i in feature_indices:
        active = (Z[:, i] > 0).astype(np.float64)
        if active.sum() == 0 or active.sum() == n:
            # All-on or all-off: no signal to correlate.
            corr = 0.0
            p_active = p_high
            p_inactive = p_high
        else:
            mu_a = active.mean()
            mu_y = y.mean()
            num = ((active - mu_a) * (y - mu_y)).mean()
            den = active.std() * y.std() + 1e-12
            corr = float(num / den)
            p_active = float(y[active > 0].mean())
            p_inactive = float(y[active == 0].mean())

        if abs(corr) < neutral_threshold:
            direction = "NEUTRAL"
        elif corr > 0:
            direction = "HIGH"
        else:
            direction = "LOW"

        out[i] = FeaturePolarity(
            feature_idx=i,
            correlation=corr,
            direction=direction,
            strength=_label_strength(corr),
            p_high_when_active=p_active,
            p_high_when_inactive=p_inactive,
        )
    return out


def render_polarity_tag(pol: FeaturePolarity) -> str:
    """Compact human-readable tag for LLM2 prompt.

    Examples:
        "[→ HIGH, strong, P(high|active)=0.83]"
        "[→ LOW, moderate, P(high|active)=0.31]"
        "[neutral]"
    """
    if pol.direction == "NEUTRAL":
        return "[neutral]"
    return (
        f"[→ {pol.direction}, {pol.strength}, "
        f"P(high|active)={pol.p_high_when_active:.2f}]"
    )

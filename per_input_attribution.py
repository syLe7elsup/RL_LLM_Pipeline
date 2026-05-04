"""Per-input feature attribution: how much does each SAE latent push the
DQN's IMV logit on THIS specific input?

Replaces (or supplements) ``feature_polarity.compute_polarities``, which is a
*marginal* statistic averaged over the whole training set. The marginal
polarity correctly captures linear / monotonic effects, but on real clinical
data the DQN can have non-monotonic combinations: a feature whose marginal
correlation with IMV is negative may still *cause* IMV in a critically-ill
patient because of how it combines with other simultaneously active features.
We saw this on three "23 LOW + 0 HIGH" patients in run_001 where every active
feature had marginal polarity → non-IMV but the model confidently predicted
IMV (P > 0.9).

Method: gradient × activation. We forward-prop the (already-encoded) SAE
latent z through the SAE decoder back into the DQN's hidden space, then
through the DQN output head, and take the gradient of the IMV logit with
respect to z. Multiplying by z gives a per-feature signed contribution.

    attribution[i]  >  0  →  this feature pushes q_imv up   (toward IMV)
    attribution[i]  <  0  →  this feature pushes q_imv down (away from IMV)
    attribution[i]  =  0  →  inactive in this input

The sign tells direction; the magnitude (relative to other active features)
tells how decisive this feature is in the model's local decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FeatureAttribution:
    """Per-feature, per-input attribution wrt the IMV logit."""

    feature_idx: int
    attribution: float          # gradient * activation, signed
    activation: float           # raw z[i], always >= 0
    direction: str              # "HIGH" (→IMV) / "LOW" (→other) / "NEUTRAL"
    strength: str               # "weak" / "moderate" / "strong" relative to top |attr|


def _strength_label(rel_magnitude: float) -> str:
    if rel_magnitude < 0.2:
        return "weak"
    if rel_magnitude < 0.6:
        return "moderate"
    return "strong"


def compute_attribution(
    qnet,
    sae,
    z_input: np.ndarray,
    *,
    imv_action: int = 2,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute gradient × activation on q_imv wrt SAE latents.

    Args:
        qnet:       LLM_RL QNetwork (input -> 128 -> 64 -> 3 actions).
        sae:        LLM_RL SparseAutoencoder (with .decoder).
        z_input:    (latent_dim,) sparse SAE latent for the input. Must be
                    the latent that came from this input's hidden state.
        imv_action: which output index corresponds to IMV (default 2).

    Returns:
        attribution (latent_dim,)  — float array, sign × magnitude per feature.
    """
    if device is None:
        device = next(qnet.parameters()).device

    z = torch.from_numpy(z_input.astype(np.float32)).to(device).unsqueeze(0)  # (1, L)
    z.requires_grad_(True)

    # Reconstruct hidden state from SAE latent, then push through DQN's output head.
    # NOTE: we use h_hat (decoded), not the original hidden state h, so that the
    # gradient flows entirely through z. h_hat ≈ h (recon loss is small ~0.017).
    h_hat = sae.decoder(z)             # (1, hidden)
    q     = qnet.out(h_hat)            # (1, n_actions)
    q_imv = q[0, imv_action]
    q_imv.backward()

    grad = z.grad.detach().cpu().numpy().reshape(-1)         # (L,)
    z_np = z.detach().cpu().numpy().reshape(-1)
    return (grad * z_np).astype(np.float32)


def attributions_to_features(
    attribution: np.ndarray,
    active_indices: list[int],
    *,
    neutral_threshold_rel: float = 0.05,
) -> dict[int, FeatureAttribution]:
    """Wrap raw attribution numbers into ``FeatureAttribution`` objects, with
    direction (HIGH/LOW/NEUTRAL) and strength (weak/moderate/strong) derived
    relative to the top |attribution| within this input.
    """
    if not active_indices:
        return {}
    abs_max = max(abs(float(attribution[i])) for i in active_indices) + 1e-12
    out: dict[int, FeatureAttribution] = {}
    for i in active_indices:
        attr = float(attribution[i])
        rel = abs(attr) / abs_max
        if rel < neutral_threshold_rel:
            direction = "NEUTRAL"
        elif attr > 0:
            direction = "HIGH"
        else:
            direction = "LOW"
        out[i] = FeatureAttribution(
            feature_idx=i,
            attribution=attr,
            activation=float(np.abs(attribution[i] / (1e-12 + np.sign(attr)))),
            direction=direction,
            strength=_strength_label(rel),
        )
    return out


def render_attribution_tag(fa: FeatureAttribution) -> str:
    """LLM2 prompt tag — input-specific, drop-in replacement for
    ``feature_polarity.render_polarity_tag``.
    """
    if fa.direction == "NEUTRAL":
        return "[neutral, contribution≈0]"
    return (
        f"[→ {'IMV' if fa.direction == 'HIGH' else 'other'}, "
        f"{fa.strength}, contribution={fa.attribution:+.3f}]"
    )


def attribution_to_polarity_proxy(
    attribution: np.ndarray,
    active_indices: list[int],
    *,
    neutral_threshold_rel: float = 0.05,
):
    """Build a dict that *quacks like* ``{feature_idx: FeaturePolarity}`` but
    derived from per-input attribution instead of marginal correlation.

    This lets us swap attributions into the existing pipeline without
    touching ``llm2_explainer`` or ``pipeline.explain_with_selection``: they
    both consume the polarities dict only via ``.direction`` /
    ``.p_high_when_active`` attributes.
    """
    from .feature_polarity import FeaturePolarity

    fa_dict = attributions_to_features(
        attribution, active_indices, neutral_threshold_rel=neutral_threshold_rel,
    )
    if not fa_dict:
        return {}
    abs_max = max(abs(fa.attribution) for fa in fa_dict.values()) + 1e-12
    out = {}
    for i, fa in fa_dict.items():
        # Map attribution magnitude to a [0, 1] "p_high_when_active" surrogate
        # so the existing polarity tag formatter prints something sensible:
        #    0.5 → neutral, 1.0 → strongly HIGH, 0.0 → strongly LOW
        rel = fa.attribution / abs_max
        p_high = float(0.5 + 0.5 * rel)            # in [0, 1]
        out[i] = FeaturePolarity(
            feature_idx=i,
            correlation=float(fa.attribution),     # repurposed: per-input attr
            direction=fa.direction,
            strength=fa.strength,
            p_high_when_active=p_high,
            p_high_when_inactive=0.5,              # not meaningful per-input
        )
    return out

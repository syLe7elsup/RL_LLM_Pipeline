"""Evidence-set collection for SAE features.

For each SAE latent feature i, we maintain:
    E_i^+   :   examples that strongly activate feature i (top activations)
    E_i^-   :   examples that do NOT activate feature i (near-zero)

We split these into a training set (shown to LLM1 when proposing the concept)
and a held-out set (used by the verifier to score Acc(c_i)).

An "example" here is a row of the *named feature vector* X (28-dim, see
``features.py``). When we hand it to LLM1 we render it as a short
human-readable bullet list of the named features so the LLM can reason about
*what kind of trajectory* activates this latent.

This module is data-only (no LLM calls). The rendering helper produces the
text strings that ``llm1_concept.py`` and ``verifier.py`` consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureEvidence:
    feature_idx: int
    pos_train_idx: np.ndarray   # indices into X with high activation, shown to LLM1
    neg_train_idx: np.ndarray   # indices into X with ~zero activation, shown to LLM1
    pos_val_idx: np.ndarray     # held-out high-activation indices for verifier
    neg_val_idx: np.ndarray     # held-out near-zero indices for verifier
    pos_act_train: np.ndarray   # corresponding activation values
    pos_act_val: np.ndarray
    density: float              # frac of all rows that activate this feature


def collect_evidence(
    Z: np.ndarray,
    *,
    n_pos: int = 8,
    n_neg: int = 8,
    n_pos_val: int = 16,
    n_neg_val: int = 16,
    min_density: float = 0.005,
    max_density: float = 0.95,
    seed: int = 0,
) -> dict[int, FeatureEvidence]:
    """Build evidence sets for every SAE feature with density in [min, max].

    Args:
        Z:        (n, latent_dim) sparse latent matrix from ``sae.encode_all``.
        n_pos:    examples in the LLM1-facing positive set per feature.
        n_neg:    examples in the LLM1-facing negative set per feature.
        n_pos_val: held-out positive examples for the verifier.
        n_neg_val: held-out negative examples for the verifier.
        min_density: skip features that fire on fewer than this fraction of rows
            (they are dead or near-dead — not enough positives to label).
        max_density: skip features that fire on MORE than this fraction. A
            feature that's "always on" is a constant and contributes no
            discriminative signal; near-saturated features (e.g. density 0.97)
            are also typically uninformative and produce too few negatives.

    Returns:
        ``{feature_idx: FeatureEvidence}`` for each feature kept.
    """
    rng = np.random.default_rng(seed)
    n_rows, latent_dim = Z.shape
    density = (Z > 0).mean(axis=0)

    out = {}
    needed_pos = n_pos + n_pos_val
    needed_neg = n_neg + n_neg_val

    for i in range(latent_dim):
        if density[i] < min_density or density[i] > max_density:
            continue
        acts = Z[:, i]
        pos_idx_all = np.where(acts > 0)[0]
        neg_idx_all = np.where(acts == 0)[0]

        if len(pos_idx_all) < needed_pos:
            # Feature fires too rarely to provide an honest val split; skip.
            continue
        if len(neg_idx_all) < needed_neg:
            continue

        # Top by activation value, then split into train/val deterministically
        # (top-half for train evidence, mid-tier for val) so the LLM is shown
        # the strongest positives and the verifier sees still-strong-but-unseen ones.
        pos_sorted = pos_idx_all[np.argsort(-acts[pos_idx_all])]
        pos_train_idx = pos_sorted[:n_pos]
        pos_val_idx = pos_sorted[n_pos : n_pos + n_pos_val]

        # Negatives: random sample (all are exactly zero so order is meaningless)
        neg_pick = rng.choice(neg_idx_all, size=needed_neg, replace=False)
        neg_train_idx = neg_pick[:n_neg]
        neg_val_idx = neg_pick[n_neg : n_neg + n_neg_val]

        out[i] = FeatureEvidence(
            feature_idx=i,
            pos_train_idx=pos_train_idx,
            neg_train_idx=neg_train_idx,
            pos_val_idx=pos_val_idx,
            neg_val_idx=neg_val_idx,
            pos_act_train=acts[pos_train_idx],
            pos_act_val=acts[pos_val_idx],
            density=float(density[i]),
        )
    return out


def render_example(x_row: np.ndarray, feature_names: list[str], top_k: int = 8) -> str:
    """Render a single named-feature vector as a compact bullet list.

    We highlight the top-k features (by absolute deviation from the per-feature
    median, so booleans show up when they're 1 in a context where median is 0)
    plus the values of all binary treatment-pattern flags so the LLM can see
    treatment context. This is what gets shown to LLM1 / verifier.
    """
    pieces = []
    abs_dev = np.abs(x_row)  # raw value rank works well for [0,1] indicators + 0/1 flags
    top_idx = np.argsort(-abs_dev)[:top_k]
    seen = set()
    for j in top_idx:
        name = feature_names[j]
        seen.add(j)
        v = float(x_row[j])
        pieces.append(f"  - {name}={v:+.2f}")
    return "\n".join(pieces)


def render_evidence_block(
    X: np.ndarray,
    feature_names: list[str],
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    label_pos: str = "POSITIVE (feature strongly active)",
    label_neg: str = "NEGATIVE (feature near zero)",
    max_lines: int | None = None,
) -> str:
    """Render side-by-side positive and negative evidence as text for the LLM."""
    def _section(label, idxs):
        lines = [f"### {label}"]
        for k, j in enumerate(idxs[: max_lines or len(idxs)]):
            lines.append(f"Example {k+1}:")
            lines.append(render_example(X[j], feature_names))
        return "\n".join(lines)

    return _section(label_pos, pos_idx) + "\n\n" + _section(label_neg, neg_idx)

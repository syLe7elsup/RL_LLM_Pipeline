"""End-to-end orchestration: pick K candidate explanations for a single input,
score each with the Judge against the blackbox prediction, and return the
winning explanation. This is Step 6 of the PDF (selection + gold buffer).

The "K candidates" in the PDF come from running LLM1 multiple times to
produce different concept dictionaries. We approximate this two ways:

1. Without polarities: temperature variation across the K LLM2 calls.
2. With polarities: in addition to the free-choice candidate, we *prescribe*
   different ``S`` subsets — top-LOW-only, top-HIGH-only, or balanced —
   so the K candidates span the polarity space. This solves the "LLM2
   cherry-picks one polarity" failure mode observed when the input has
   both LOW- and HIGH-pushing features active.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .judge import JudgePrediction, judge_explanation, kl_blackbox_judge
from .llm import LLMClient
from .llm2_explainer import StructuredExplanation, explain_one


@dataclass
class Candidate:
    explanation: StructuredExplanation
    judge: JudgePrediction
    kl: float
    sparsity: int       # |S|
    length: int         # number of mechanism claims
    total_loss: float
    strategy: str = "free"  # "free", "low_only", "high_only", "balanced"


@dataclass
class ExplanationResult:
    input_idx: int
    p_blackbox: np.ndarray
    candidates: list[Candidate]
    winner_index: int

    @property
    def winner(self) -> Candidate:
        return self.candidates[self.winner_index]


def _candidate_strategies(
    active_features: list[int],
    activation_values: list[float],
    polarities: dict | None,
    K: int,
    top_per_side: int = 2,
):
    """Decide each candidate's S-prescription strategy.

    Returns a list of (strategy_name, forced_S_or_None, temperature) tuples
    of length K. Strategies depend on whether polarity info is available
    and whether the active features split into both LOW and HIGH polarities.
    """
    # Default strategy (no polarities): temperature variation only.
    if polarities is None:
        return [("free", None, 0.0)] + [("free", None, 0.7)] * (K - 1)

    # Sort active features by activation, broken down by polarity.
    by_act = sorted(
        zip(active_features, activation_values),
        key=lambda x: -x[1],
    )
    low = [(f, v) for f, v in by_act if f in polarities and polarities[f].direction == "LOW"]
    high = [(f, v) for f, v in by_act if f in polarities and polarities[f].direction == "HIGH"]

    if not low or not high:
        # Single polarity (or all neutral): vanilla temperature sweep is fine.
        return [("free", None, 0.0)] + [("free", None, 0.7)] * (K - 1)

    # Mixed polarity: prescribe diverse S subsets.
    low_top = [f for f, _ in low[:top_per_side]]
    high_top = [f for f, _ in high[:top_per_side]]
    balanced = [low[0][0], high[0][0]]

    # Order matters: balanced is usually the winner in mixed cases, so put it
    # second (after free) — that way even K=3 still gets the most useful
    # diversification.
    strategies = [
        ("free", None, 0.0),
        ("balanced", balanced, 0.0),
        ("low_only", low_top, 0.0),
        ("high_only", high_top, 0.0),
    ]
    return strategies[:K]


def explain_with_selection(
    client: LLMClient,
    input_idx: int,
    *,
    p_blackbox: np.ndarray,
    active_features: list[int],
    activation_values: list[float],
    concept_labels: dict[int, str],
    polarities: dict | None = None,
    K: int = 3,
    temperatures: list[float] | None = None,
    lambda_spar: float = 0.1,
    lambda_len: float = 0.05,
) -> ExplanationResult:
    """Generate K candidate explanations, score them, return all + the winner.

    Total loss per candidate (Step 6 of the PDF):
        L = KL(P_M || P_Judge) + lambda_spar * |S| + lambda_len * Len(E)

    If ``polarities`` is provided AND the active features include both LOW-
    and HIGH-pushing concepts, the K candidates are diversified by
    prescribing different ``S`` subsets (free / low-only / high-only /
    balanced). This prevents the "cherry-pick one polarity" failure mode.
    Otherwise we fall back to plain temperature variation.
    """
    strategies = _candidate_strategies(
        active_features, activation_values, polarities, K
    )
    if temperatures is not None and len(temperatures) == K:
        # Caller-provided temperatures override per-strategy defaults.
        strategies = [(name, s, t) for (name, s, _), t in zip(strategies, temperatures)]

    cands: list[Candidate] = []
    for strategy_name, forced_S, t in strategies:
        expl = explain_one(
            client,
            active_features=active_features,
            activation_values=activation_values,
            concept_labels=concept_labels,
            polarities=polarities,
            forced_S=forced_S,
            temperature=t,
        )
        jp = judge_explanation(client, expl)
        kl = kl_blackbox_judge(p_blackbox, jp.probs)
        sparsity = len(expl.cited_features)
        length = len(expl.mechanisms)
        total = kl + lambda_spar * sparsity + lambda_len * length
        cands.append(
            Candidate(
                explanation=expl,
                judge=jp,
                kl=kl,
                sparsity=sparsity,
                length=length,
                total_loss=total,
                strategy=strategy_name,
            )
        )

    winner = int(np.argmin([c.total_loss for c in cands]))
    return ExplanationResult(
        input_idx=input_idx,
        p_blackbox=np.asarray(p_blackbox, dtype=np.float64),
        candidates=cands,
        winner_index=winner,
    )

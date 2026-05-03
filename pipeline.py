"""End-to-end orchestration: pick K candidate explanations for a single input,
score each with the Judge against the blackbox prediction, and return the
winning explanation. This is Step 6 of the PDF (selection + gold buffer).

The "K candidates" in the PDF come from running LLM1 multiple times to
produce different concept dictionaries. In our MVP we sample K LLM2 outputs
under different temperatures (cheaper than re-running LLM1 with multiple
seeds) — same idea, fewer LLM calls.
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


@dataclass
class ExplanationResult:
    input_idx: int
    p_blackbox: np.ndarray
    candidates: list[Candidate]
    winner_index: int

    @property
    def winner(self) -> Candidate:
        return self.candidates[self.winner_index]


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
    """
    if temperatures is None:
        # First candidate is greedy (deterministic); rest are sampled.
        temperatures = [0.0] + [0.7] * (K - 1)
    if len(temperatures) != K:
        raise ValueError("temperatures must have length K")

    cands: list[Candidate] = []
    for t in temperatures:
        expl = explain_one(
            client,
            active_features=active_features,
            activation_values=activation_values,
            concept_labels=concept_labels,
            polarities=polarities,
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
            )
        )

    winner = int(np.argmin([c.total_loss for c in cands]))
    return ExplanationResult(
        input_idx=input_idx,
        p_blackbox=np.asarray(p_blackbox, dtype=np.float64),
        candidates=cands,
        winner_index=winner,
    )

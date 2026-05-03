"""Verifier: judge whether a proposed concept holds for a given trajectory.

Implements ``Ver(c_i, x) -> {0, 1}`` from the PDF (Step 3). We expose two
backends (the PDF's two practical instantiations):

- ``LLMVerifier`` — a short, constrained prompt that takes ``(concept, example)``
  and outputs only MATCH or NO_MATCH (Qwen / DashScope / etc. via ``LLMClient``).
- ``EmbeddingVerifier`` — embed the concept text and a natural-language
  rendering of the example with sentence-transformers, threshold the cosine
  similarity. Calibrates the threshold *per concept* on the train evidence
  so we don't have to pick a global threshold.

Both implement the same callable interface ``__call__(concept, x_row, names) -> 0|1``.

The verifier is then used to score concept quality on held-out evidence:
    Acc(c_i) = mean over E_i^+ ∪ E_i^- of [Ver(c_i, x) == 1[x in E_i^+]]

False-negative and false-positive indices are returned alongside the score so
``llm1_concept.refine_concept`` can use them to build a tightened prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .evidence import FeatureEvidence, render_example
from .llm import LLMClient


VERIFIER_SYSTEM = """You are a verifier. Given a CONCEPT (a short description
of a trajectory pattern) and a single EXAMPLE (a list of named feature values
from one patient trajectory), decide whether the concept holds for the example.

Reply with EXACTLY one token: MATCH or NO_MATCH. Do not explain.
"""


VERIFIER_USER_TEMPLATE = """CONCEPT: {concept}

EXAMPLE:
{example_block}

Does the concept hold for this example? Answer MATCH or NO_MATCH."""


def _parse_match(text: str) -> int:
    """Lenient parse: 1 for MATCH, 0 otherwise. We strip and uppercase, then
    check the first whitespace-delimited token."""
    t = text.strip().upper()
    first = t.split()[0] if t else ""
    return 1 if first.startswith("MATCH") else 0


def verify_one(
    client: LLMClient,
    concept: str,
    x_row: np.ndarray,
    feature_names: list[str],
    *,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
) -> int:
    user = VERIFIER_USER_TEMPLATE.format(
        concept=concept, example_block=render_example(x_row, feature_names)
    )
    text = client.chat(
        [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _parse_match(text)


# ---------------------------------------------------------------------------
# Verifier interface and two backends


class Verifier(Protocol):
    """Callable: (concept, x_row, feature_names) -> {0, 1}."""

    def __call__(self, concept: str, x_row: np.ndarray, feature_names: list[str]) -> int: ...

    def calibrate(
        self,
        concept: str,
        X: np.ndarray,
        feature_names: list[str],
        pos_idx: np.ndarray,
        neg_idx: np.ndarray,
    ) -> None: ...


@dataclass
class LLMVerifier:
    """LLM MATCH/NO_MATCH backend. Stateless — calibrate is a no-op."""

    client: LLMClient
    max_new_tokens: int = 4
    temperature: float = 0.0

    def __call__(self, concept: str, x_row: np.ndarray, feature_names: list[str]) -> int:
        return verify_one(
            self.client, concept, x_row, feature_names,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )

    def calibrate(self, concept, X, feature_names, pos_idx, neg_idx):
        return  # no-op


def _natural_language_render(x_row: np.ndarray, feature_names: list[str], top_k: int = 8) -> str:
    """Render a feature vector as a single-line natural-language description.

    Embedding models do best with prose-like input. We translate the top-k
    most salient feature=value pairs into short phrases and join with commas.
    Example output:
        "treatment_count is 1, treatment given at stage 3, indicator 2 high
         at stage 3 (0.87), indicator 1 high at stage 3 (0.86), ..."
    """
    abs_dev = np.abs(x_row)
    top_idx = np.argsort(-abs_dev)[:top_k]
    phrases = []
    for j in top_idx:
        name = feature_names[j]
        v = float(x_row[j])
        # Try to make booleans / counts read naturally.
        if name in ("a1", "a2", "a3"):
            stage = name[1]
            phrases.append(f"treatment given at stage {stage}" if v > 0.5 else f"no treatment at stage {stage}")
        elif name == "treatment_count":
            phrases.append(f"treatment count is {int(v)}")
        elif name == "switch_count":
            phrases.append(f"switched treatment {int(v)} times")
        elif name in ("all_treated", "never_treated", "early_treated"):
            if v > 0.5:
                phrases.append(name.replace("_", " "))
        elif name.startswith("delta"):
            sign = "increased" if v > 0 else "decreased"
            phrases.append(f"{name} {sign} by {abs(v):.2f}")
        else:
            level = "high" if v > 0.6 else ("medium" if v > 0.3 else "low")
            phrases.append(f"{name} {level} ({v:.2f})")
    return ", ".join(phrases)


@dataclass
class EmbeddingVerifier:
    """Cosine-similarity backend using sentence-transformers.

    The concept is embedded as-is. Each example is rendered as natural
    language (see ``_natural_language_render``) and embedded the same way.
    Threshold is calibrated *per concept* on the train evidence: we sweep
    candidate thresholds and pick the one that maximises train accuracy.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.3              # default until calibrated
    _cached_threshold: dict = field(default_factory=dict)
    _model: object | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def _cos(self, concept: str, text: str) -> float:
        self._ensure_loaded()
        embs = self._model.encode([concept, text], convert_to_numpy=True, normalize_embeddings=True)
        return float(embs[0] @ embs[1])

    def __call__(self, concept: str, x_row: np.ndarray, feature_names: list[str]) -> int:
        text = _natural_language_render(x_row, feature_names)
        thr = self._cached_threshold.get(concept, self.threshold)
        return int(self._cos(concept, text) >= thr)

    def calibrate(
        self,
        concept: str,
        X: np.ndarray,
        feature_names: list[str],
        pos_idx: np.ndarray,
        neg_idx: np.ndarray,
    ) -> None:
        """Sweep thresholds in [0, 1] on the train evidence; cache the best one."""
        self._ensure_loaded()
        pos_texts = [_natural_language_render(X[j], feature_names) for j in pos_idx]
        neg_texts = [_natural_language_render(X[j], feature_names) for j in neg_idx]
        all_texts = [concept] + pos_texts + neg_texts
        embs = self._model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
        c = embs[0]
        sims_pos = embs[1 : 1 + len(pos_idx)] @ c
        sims_neg = embs[1 + len(pos_idx) :] @ c
        best_acc, best_thr = -1.0, self.threshold
        for thr in np.linspace(-0.2, 0.9, 56):
            acc = ((sims_pos >= thr).sum() + (sims_neg < thr).sum()) / (len(pos_idx) + len(neg_idx))
            if acc > best_acc:
                best_acc, best_thr = float(acc), float(thr)
        self._cached_threshold[concept] = best_thr


@dataclass
class VerifierResult:
    accuracy: float
    fn_idx: np.ndarray   # held-out positives that the verifier missed
    fp_idx: np.ndarray   # held-out negatives that the verifier wrongly accepted
    pos_predictions: np.ndarray
    neg_predictions: np.ndarray


def score_concept(
    verifier: Verifier,
    concept: str,
    evidence: FeatureEvidence,
    X: np.ndarray,
    feature_names: list[str],
) -> VerifierResult:
    """Run the (already-calibrated) verifier on every held-out example.

    For each x in E_i^+, we want Ver(c_i, x) == 1. For each x in E_i^-,
    we want Ver(c_i, x) == 0. Accuracy is the fraction of correct answers.
    """
    pos_pred = np.array(
        [verifier(concept, X[j], feature_names) for j in evidence.pos_val_idx]
    )
    neg_pred = np.array(
        [verifier(concept, X[j], feature_names) for j in evidence.neg_val_idx]
    )
    correct = pos_pred.sum() + (1 - neg_pred).sum()
    total = len(pos_pred) + len(neg_pred)
    acc = float(correct) / max(total, 1)
    fn_mask = pos_pred == 0
    fp_mask = neg_pred == 1
    return VerifierResult(
        accuracy=acc,
        fn_idx=evidence.pos_val_idx[fn_mask],
        fp_idx=evidence.neg_val_idx[fp_mask],
        pos_predictions=pos_pred,
        neg_predictions=neg_pred,
    )


def label_with_refinement(
    client: LLMClient,
    evidence: FeatureEvidence,
    X: np.ndarray,
    feature_names: list[str],
    *,
    verifier: Verifier | None = None,
    alpha: float = 0.7,
    max_rounds: int = 1,
    propose_fn=None,
    refine_fn=None,
    gold_buffer=None,
    gold_k: int = 2,
    gold_seed: int | None = None,
):
    """End-to-end loop for one feature: propose -> calibrate -> verify -> (refine -> verify) ...

    Args:
        client:      LLM used for *concept naming* (LLM1).
        verifier:    The verifier backend (``LLMVerifier`` or ``EmbeddingVerifier``).
                     Defaults to ``LLMVerifier(client)`` to preserve old behavior.
        gold_buffer: Optional ``GoldBuffer`` of high-quality past concepts.
                     If provided and non-empty, ``gold_k`` worked examples are
                     prepended as few-shot demos in both propose and refine calls.

    Returns ``(ConceptResult, VerifierResult)``.
    """
    from .llm1_concept import ConceptResult  # local import to avoid cycle
    if propose_fn is None or refine_fn is None:
        from .llm1_concept import propose_concept as propose_fn  # type: ignore
        from .llm1_concept import refine_concept as refine_fn    # type: ignore
    if verifier is None:
        verifier = LLMVerifier(client)

    concept, alts = propose_fn(
        client, evidence, X, feature_names,
        gold_buffer=gold_buffer, gold_k=gold_k, gold_seed=gold_seed,
    )
    verifier.calibrate(concept, X, feature_names, evidence.pos_train_idx, evidence.neg_train_idx)
    last = score_concept(verifier, concept, evidence, X, feature_names)
    history = [{"round": 0, "concept": concept, "alts": alts, "accuracy": last.accuracy}]

    rounds = 0
    while last.accuracy < alpha and rounds < max_rounds:
        rounds += 1
        concept, alts = refine_fn(
            client, evidence, X, feature_names,
            current_concept=concept,
            fn_idx=last.fn_idx, fp_idx=last.fp_idx, accuracy=last.accuracy,
            gold_buffer=gold_buffer, gold_k=gold_k, gold_seed=gold_seed,
        )
        verifier.calibrate(concept, X, feature_names, evidence.pos_train_idx, evidence.neg_train_idx)
        last = score_concept(verifier, concept, evidence, X, feature_names)
        history.append(
            {"round": rounds, "concept": concept, "alts": alts, "accuracy": last.accuracy}
        )

    res = ConceptResult(
        feature_idx=evidence.feature_idx,
        concept=concept,
        alternatives=alts,
        refinement_history=history,
        final_accuracy=last.accuracy,
    )
    return res, last

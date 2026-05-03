"""Gold buffer of high-quality (concept, evidence) records used as few-shot
examples in subsequent LLM1 calls.

Mirrors the PDF Step 6 final paragraph:

    > In subsequent iterations, we update LLM1's context using a small number
    > of stored successes as few-shot demonstrations. ... By repeatedly
    > conditioning on these high-quality mappings, LLM1 becomes more stable
    > and consistent in naming features...

In our pipeline we trigger storage when a freshly-proposed concept passes the
verifier accuracy threshold (alpha). The buffer is FIFO-bounded; sampling is
deterministic given a seed (so notebook reruns are reproducible).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .evidence import FeatureEvidence, render_evidence_block


@dataclass
class GoldExample:
    feature_idx: int
    pos_idx: np.ndarray   # train-side positive indices into X
    neg_idx: np.ndarray   # train-side negative indices into X
    concept: str
    accuracy: float       # the verifier-acc that earned the spot in the buffer


@dataclass
class GoldBuffer:
    """FIFO gold-record buffer with deterministic few-shot sampling."""

    max_size: int = 10
    examples: list[GoldExample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def push(self, evidence: FeatureEvidence, concept: str, accuracy: float) -> None:
        ex = GoldExample(
            feature_idx=evidence.feature_idx,
            pos_idx=np.asarray(evidence.pos_train_idx),
            neg_idx=np.asarray(evidence.neg_train_idx),
            concept=concept,
            accuracy=accuracy,
        )
        self.examples.append(ex)
        if len(self.examples) > self.max_size:
            # FIFO eviction.
            self.examples.pop(0)

    def sample(self, k: int, seed: int | None = None) -> list[GoldExample]:
        if not self.examples:
            return []
        rng = random.Random(seed)
        if k >= len(self.examples):
            return list(self.examples)
        # Bias toward higher-accuracy examples without ignoring diversity:
        # weight by accuracy^2.
        weights = [max(ex.accuracy, 0.0) ** 2 + 1e-6 for ex in self.examples]
        return rng.choices(self.examples, weights=weights, k=k)

    def render_few_shot(
        self,
        X: np.ndarray,
        feature_names: list[str],
        k: int = 2,
        max_pos_per_example: int = 2,
        max_neg_per_example: int = 2,
        seed: int | None = None,
    ) -> str:
        """Render up to ``k`` buffered examples as few-shot demonstrations.

        Empty string if the buffer is empty (so callers can unconditionally
        prepend the result without branching).
        """
        picks = self.sample(k, seed=seed)
        if not picks:
            return ""
        blocks = []
        for ex in picks:
            ev_text = render_evidence_block(
                X, feature_names,
                ex.pos_idx[:max_pos_per_example],
                ex.neg_idx[:max_neg_per_example],
            )
            blocks.append(
                f"### Worked example (feature i{ex.feature_idx}, verifier acc={ex.accuracy:.2f}):\n"
                f"{ev_text}\n"
                f'Correct concept: "{ex.concept}"'
            )
        header = (
            "Below are worked examples of well-named features from previous rounds. "
            "Use them as a guide for the *style* and *specificity* of a good concept "
            "(short, observable, treatment- and indicator-grounded), then propose a "
            "concept for the new feature.\n"
        )
        return header + "\n\n".join(blocks) + "\n\n"

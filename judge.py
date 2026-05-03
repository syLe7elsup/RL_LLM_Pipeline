"""Judge: predict P(class | explanation) using *only* the structured
explanation as input (no access to the original trajectory).

This is Step 5 of the PDF pipeline. The Judge emits a JSON object with
``prob_high`` and ``prob_low`` so we can compute the predictive KL loss
between blackbox M and Judge.

We take care to:
    - Constrain the output format (JSON with two probabilities).
    - Fall back to a uniform prior if parsing fails (so KL never blows up).
    - Never show the Judge anything except the explanation text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .llm import LLMClient
from .llm2_explainer import StructuredExplanation


JUDGE_SYSTEM = """You are a judge. Given ONLY a short structured explanation of
why a model made its prediction (you do NOT see the original input), output a
probability distribution over the two possible outcomes:

  - HIGH outcome (label = 1)
  - LOW outcome  (label = 0)

The explanation cites SAE features by [cite: i_k]. Use the explanation's
content (mechanisms, aggregation, limits) to weigh how strongly each outcome
is supported.

Reply with EXACTLY one JSON object on a single line:
{"prob_high": <float in [0,1]>, "prob_low": <float in [0,1]>}
The two probabilities must sum to 1. Do not include any other text.
"""


JUDGE_USER_TEMPLATE = """Structured explanation:
{explanation_text}

Output the JSON now."""


_JSON_RE = re.compile(r"\{[^{}]*\"prob_high\"[^{}]*\}", re.DOTALL)


def _parse_judge(text: str) -> tuple[float, float]:
    """Lenient JSON parse. Returns (prob_low, prob_high) — sums to 1."""
    m = _JSON_RE.search(text)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
        ph = float(obj.get("prob_high", 0.5))
        pl = float(obj.get("prob_low", 1.0 - ph))
        # Renormalize defensively (in case the model emitted unnormalized values).
        s = ph + pl
        if s <= 0:
            return 0.5, 0.5
        return pl / s, ph / s
    except Exception:
        return 0.5, 0.5


@dataclass
class JudgePrediction:
    prob_low: float
    prob_high: float
    raw_text: str

    @property
    def probs(self) -> np.ndarray:
        return np.array([self.prob_low, self.prob_high], dtype=np.float64)


def judge_explanation(
    client: LLMClient,
    explanation: StructuredExplanation,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> JudgePrediction:
    text = client.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    explanation_text=explanation.raw_text
                ),
            },
        ],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    pl, ph = _parse_judge(text)
    return JudgePrediction(prob_low=pl, prob_high=ph, raw_text=text)


def kl_blackbox_judge(p_blackbox: np.ndarray, p_judge: np.ndarray, eps: float = 1e-8) -> float:
    """KL(P_M || P_Judge), the predictive loss in the PDF.

    Both inputs should be 1-D arrays [p_low, p_high] summing to 1.
    """
    p_m = np.asarray(p_blackbox, dtype=np.float64) + eps
    p_j = np.asarray(p_judge, dtype=np.float64) + eps
    p_m = p_m / p_m.sum()
    p_j = p_j / p_j.sum()
    return float((p_m * np.log(p_m / p_j)).sum())

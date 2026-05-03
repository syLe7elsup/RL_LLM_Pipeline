"""LLM1: propose human-readable concept labels for SAE features.

Mirrors Step 2 + Step 3 (refinement) of the PDF pipeline:
    - Show LLM1 the evidence sets (E_i^+, E_i^-).
    - Ask for a concise concept phrase.
    - Verify on held-out evidence (see ``verifier.py``).
    - If accuracy is below ``alpha``, build a refinement prompt that includes
      sampled false negatives / false positives and ask LLM1 to revise.
    - Repeat for at most ``max_rounds`` rounds.

The prompts here are adapted to our DTR/trajectory setting (instead of
images): "concepts" are descriptions of trajectory patterns, e.g. "high
stage-3 indicators with consistent treatment".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .evidence import FeatureEvidence, render_evidence_block
from .llm import LLMClient


LLM1_SYSTEM = """You are a concept-namer labeling a Sparse Autoencoder feature using evidence.

Your goal is to produce a short, human-readable concept that best separates
POSITIVE (high-activation) trajectory examples from NEGATIVE (near-zero) ones.

Rules:
1) The concept must be directly observable in the examples — describe a
   pattern in indicator values, deltas, or treatment choices that is true of
   the positives and absent in the negatives.
2) Do NOT mention the predicted outcome (high/low) or any class label.
3) Output a concise phrase, at most 8 words.
4) If the feature appears polysemantic, output MIXED and propose two candidate
   concepts.
5) Output exactly the required fields, nothing else.
"""


LLM1_USER_TEMPLATE = """Feature index: {feature_idx}

Each example below is a trajectory through stages 1..3. We show the most
salient feature names with their values. Indicator values are in [0,1];
treatment flags are 0/1; deltas can be negative.

{evidence_block}

Task: Propose a concise concept (at most 8 words) that distinguishes the
POSITIVES from the NEGATIVES.

Output format (exactly these fields):
Concept: <at most 8 words>
Alt concepts: <up to 2 short alternatives, or NA>"""


REFINE_USER_TEMPLATE = """Feature index: {feature_idx}
Current concept: {current_concept}
Held-out verifier accuracy: {acc:.2f}   FN count: {n_fn}   FP count: {n_fp}

False negatives (verifier said NO_MATCH but examples were positive):
{fn_block}

False positives (verifier said MATCH but examples were negative):
{fp_block}

Task: Propose a revised concept that (i) covers the FN examples and (ii)
excludes the FP examples.

Output format (exactly these fields):
Revised concept: <at most 8 words>
Inclusion cues: <1-2 cues that cover the FNs>
Exclusion cues: <1-2 cues that exclude the FPs>
Alt concepts: <up to 2 short alternatives, or NA>"""


_CONCEPT_RE = re.compile(r"^\s*(?:Revised concept|Concept)\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_ALT_RE = re.compile(r"^\s*Alt concepts?\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_concept_response(text: str) -> tuple[str, list[str]]:
    """Pull the concept phrase and any alternatives out of LLM1's reply.

    Returns ``(concept, alternatives)``. If parsing fails, returns the whole
    text as the concept (so we still have a label downstream) and empty alts.
    """
    m = _CONCEPT_RE.search(text)
    concept = m.group(1).strip() if m else text.strip().split("\n", 1)[0].strip()
    a = _ALT_RE.search(text)
    alts: list[str] = []
    if a:
        raw = a.group(1).strip()
        if raw.upper() != "NA":
            alts = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
    return concept, alts


@dataclass
class ConceptResult:
    feature_idx: int
    concept: str
    alternatives: list[str] = field(default_factory=list)
    refinement_history: list[dict] = field(default_factory=list)
    final_accuracy: float | None = None


def propose_concept(
    client: LLMClient,
    evidence: FeatureEvidence,
    X: np.ndarray,
    feature_names: list[str],
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    gold_buffer=None,
    gold_k: int = 2,
    gold_seed: int | None = None,
) -> tuple[str, list[str]]:
    """Single-shot LLM1 call: propose a concept from the *training* evidence.

    If ``gold_buffer`` is provided and non-empty, ``gold_k`` worked examples
    are prepended as few-shot demonstrations (PDF Step 6 final paragraph).
    """
    block = render_evidence_block(
        X, feature_names, evidence.pos_train_idx, evidence.neg_train_idx
    )
    user_body = LLM1_USER_TEMPLATE.format(
        feature_idx=evidence.feature_idx, evidence_block=block
    )
    few_shot = ""
    if gold_buffer is not None and len(gold_buffer) > 0:
        few_shot = gold_buffer.render_few_shot(
            X, feature_names, k=gold_k, seed=gold_seed
        )
    text = client.chat(
        [
            {"role": "system", "content": LLM1_SYSTEM},
            {"role": "user", "content": few_shot + user_body},
        ],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _parse_concept_response(text)


def refine_concept(
    client: LLMClient,
    evidence: FeatureEvidence,
    X: np.ndarray,
    feature_names: list[str],
    current_concept: str,
    fn_idx: np.ndarray,
    fp_idx: np.ndarray,
    accuracy: float,
    *,
    max_fn_show: int = 3,
    max_fp_show: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    gold_buffer=None,
    gold_k: int = 2,
    gold_seed: int | None = None,
) -> tuple[str, list[str]]:
    """Refinement call: feed back FN/FP examples and ask for a tightened concept."""
    fn_block = "\n\n".join(
        f"FN#{k+1}:\n" + render_evidence_block(X, feature_names, np.array([j]), np.array([]))
        .replace("### POSITIVE (feature strongly active)\n", "")
        .replace("### NEGATIVE (feature near zero)\n", "")
        .strip()
        for k, j in enumerate(fn_idx[:max_fn_show])
    ) or "(none)"
    fp_block = "\n\n".join(
        f"FP#{k+1}:\n" + render_evidence_block(X, feature_names, np.array([j]), np.array([]))
        .replace("### POSITIVE (feature strongly active)\n", "")
        .replace("### NEGATIVE (feature near zero)\n", "")
        .strip()
        for k, j in enumerate(fp_idx[:max_fp_show])
    ) or "(none)"

    user_body = REFINE_USER_TEMPLATE.format(
        feature_idx=evidence.feature_idx,
        current_concept=current_concept,
        acc=accuracy,
        n_fn=len(fn_idx),
        n_fp=len(fp_idx),
        fn_block=fn_block,
        fp_block=fp_block,
    )
    few_shot = ""
    if gold_buffer is not None and len(gold_buffer) > 0:
        few_shot = gold_buffer.render_few_shot(
            X, feature_names, k=gold_k, seed=gold_seed
        )
    text = client.chat(
        [
            {"role": "system", "content": LLM1_SYSTEM},
            {"role": "user", "content": few_shot + user_body},
        ],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _parse_concept_response(text)

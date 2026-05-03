"""LLM2: produce a constrained, structured explanation citing only the
SAE concepts that are active for a given input.

Implements Step 4 of the PDF pipeline. The output schema is exactly:
    S: {i_1, i_2, ...}
    Mechanisms:
    - <claim 1> [cite: ...]
    - <claim 2> [cite: ...]
    Aggregation:
    - <how cited cues support one class vs the other> [cite: ...]
    Limits:
    - <what is not covered / uncertainty> [cite: ...]

Constraints:
    - LLM2 may only reference features in the provided concept dictionary.
    - Every claim must include at least one [cite: i_k].
    - Cited subset S is kept small (<= 3 by default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient


LLM2_SYSTEM = """You are an explainer that must be strictly faithful to a
provided set of SAE concepts about a patient trajectory.

Rules:
1) You may ONLY use the concepts provided in the concept dictionary below.
2) Every factual claim must cite at least one feature index in the format [cite: i_k].
3) Do NOT introduce external knowledge, class stereotypes, or attributes not
   present in the dictionary.
4) Output MUST follow the exact schema below and include nothing else.
5) Keep the cited feature set small (use at most 3 features unless necessary).

POLARITY TAGS:
Each concept may be annotated with a polarity tag like "[→ HIGH, strong, P(high|active)=0.83]".
This tells you the empirical direction this feature pushes the model:
  - "→ HIGH" means activating this feature is associated with HIGH outcome.
  - "→ LOW" means activating this feature is associated with LOW outcome.
  - "[neutral]" means no clear directional signal — be cautious citing it.
Your Aggregation MUST respect these tags: do NOT claim a "[→ LOW]" concept
supports HIGH, or vice versa. If polarities of cited features disagree with
each other, say so honestly (the trajectory has mixed signals).

Schema (output exactly):
S: {i_1, i_2, ...}
Mechanisms:
- <claim 1> [cite: ...]
- <claim 2> [cite: ...]
Aggregation:
- <how the cited cues support one class vs the other> [cite: ...]
Limits:
- <what is not covered / uncertainty> [cite: ...]
"""


LLM2_USER_TEMPLATE = """Task: Explain why the model predicts HIGH outcome vs LOW outcome
for this patient trajectory using ONLY the provided SAE concepts.

Active feature-value pairs (feature index : activation value):
{active_pairs}

Concept dictionary (you may use ONLY these):
{concept_dict}

Instructions:
1) Choose a small cited subset S (prefer 2 features).
2) Write short mechanism claims that reference only the dictionary concepts.
3) In Aggregation, state how these mechanisms push toward HIGH vs LOW outcome.
4) In Limits, mention that other cues may exist and that concepts are incomplete."""


@dataclass
class StructuredExplanation:
    raw_text: str
    cited_features: list[int] = field(default_factory=list)  # parsed S
    mechanisms: list[str] = field(default_factory=list)
    aggregation: str = ""
    limits: str = ""


_S_RE = re.compile(r"^\s*S\s*:\s*\{?\s*([^}\n]+?)\s*\}?\s*$", re.MULTILINE | re.IGNORECASE)
_SECTION_RE = re.compile(
    r"^(Mechanisms|Aggregation|Limits)\s*:\s*$", re.MULTILINE | re.IGNORECASE
)


def _parse_explanation(text: str) -> StructuredExplanation:
    out = StructuredExplanation(raw_text=text)
    m = _S_RE.search(text)
    if m:
        # accept both 'i_1, i_2' and '5, 12' style
        for tok in re.findall(r"i?_?(\d+)", m.group(1)):
            try:
                out.cited_features.append(int(tok))
            except ValueError:
                pass
    # Section parse: split on the section headers, keep everything after
    sections: dict[str, str] = {}
    last = None
    for line in text.splitlines():
        h = _SECTION_RE.match(line)
        if h:
            last = h.group(1).lower()
            sections[last] = ""
        elif last is not None:
            sections[last] += line + "\n"
    if "mechanisms" in sections:
        out.mechanisms = [
            ln.lstrip("- ").strip()
            for ln in sections["mechanisms"].splitlines()
            if ln.strip().startswith("-")
        ]
    out.aggregation = sections.get("aggregation", "").strip()
    out.limits = sections.get("limits", "").strip()
    return out


def explain_one(
    client: LLMClient,
    *,
    active_features: list[int],
    activation_values: list[float],
    concept_labels: dict[int, str],
    polarities: dict | None = None,
    max_new_tokens: int = 320,
    temperature: float = 0.0,
) -> StructuredExplanation:
    """Build the prompt and call LLM2 for a single input.

    Args:
        active_features:    indices of SAE features currently active for x.
        activation_values:  same length as active_features.
        concept_labels:     {feature_idx: concept string} for *all* named features
                            (we only show the active ones in the prompt body but
                            the LLM is told the full dictionary).
        polarities:         {feature_idx: FeaturePolarity} (optional). If
                            provided, each dict entry is annotated with a tag
                            like "[→ HIGH, strong, P(high|active)=0.83]" so
                            LLM2 can reason about direction faithfully.
    """
    from .feature_polarity import render_polarity_tag

    pair_lines = "\n".join(
        f"i{i}: {v:.2f}"
        for i, v in zip(active_features, activation_values)
    )

    def _dict_line(i: int) -> str:
        label = concept_labels[i]
        tag = ""
        if polarities is not None and i in polarities:
            tag = " " + render_polarity_tag(polarities[i])
        return f'i{i} -> "{label}"{tag}'

    dict_lines = "\n".join(
        _dict_line(i) for i in active_features if i in concept_labels
    )
    user = LLM2_USER_TEMPLATE.format(active_pairs=pair_lines, concept_dict=dict_lines)
    text = client.chat(
        [
            {"role": "system", "content": LLM2_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _parse_explanation(text)

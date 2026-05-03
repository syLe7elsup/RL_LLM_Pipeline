"""Pluggable LLM client used by every model-facing step of the pipeline.

Three concrete clients:
    - ``QwenLocalClient`` — runs Qwen2.5-Instruct via huggingface ``transformers``.
      Default for users with a GPU. Set ``model_id`` to swap sizes
      (e.g. ``Qwen/Qwen2.5-1.5B-Instruct`` for low memory,
      ``Qwen/Qwen2.5-7B-Instruct`` for a bigger machine).
    - ``StubClient`` — deterministic, no network/no GPU. Use it to dry-run
      the pipeline (verify wiring, schemas, evidence formatting) without
      burning real LLM time. Returns canned outputs that satisfy each
      role's parsing format.
    - ``DashScopeClient`` — optional, hits Alibaba DashScope's hosted Qwen
      via REST. Only used if ``DASHSCOPE_API_KEY`` is set.

All three implement the same minimal interface: ``chat(messages, ...) -> str``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Local Qwen via transformers


@dataclass
class QwenLocalClient:
    """Wraps a local Qwen2.5-Instruct model. Lazy-loads on first ``chat`` call."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str | None = None  # auto-detect if None: cuda > mps > cpu
    dtype: str = "auto"        # "auto", "float16", "bfloat16", "float32"
    name: str = "qwen-local"

    def __post_init__(self):
        self._tok = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if self.dtype == "auto":
            torch_dtype = torch.float16 if device != "cpu" else torch.float32
        else:
            torch_dtype = getattr(torch, self.dtype)

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype
        ).to(device)
        self._model.eval()
        self._device = device

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        self._ensure_loaded()
        import torch

        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self._tok.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        text = self._tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text.strip()


# ---------------------------------------------------------------------------
# Stub client (no LLM)


@dataclass
class StubClient:
    """Deterministic stand-in for testing the pipeline without a real LLM.

    The stub recognizes which role is calling it via a marker in the system
    message and returns a reasonable canned reply. Use this for development
    and CI to verify the pipeline plumbing end-to-end in seconds.
    """

    name: str = "stub"

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        # Concatenate role markers so we can detect which step is calling.
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

        if "concept-namer" in sys_msg or "Sparse Autoencoder feature using evidence" in sys_msg:
            # LLM1: emit a concept label + 2 alts. We hash the user msg to keep
            # outputs stable across reruns of the same evidence.
            tag = abs(hash(user_msg)) % 5
            label = [
                "high stage-3 indicators",
                "consistent treatment with rising indicators",
                "early treatment then plateau",
                "indicator decline despite treatment",
                "low indicators throughout",
            ][tag]
            return (
                f"Concept: {label}\n"
                f"Alt concepts: high s3_ind1; high s3_ind2"
            )

        if "verifier" in sys_msg or sys_msg.strip().endswith("MATCH or NO_MATCH"):
            # Deterministic but evidence-aware: positive examples in the prompt
            # tend to mention "POSITIVE" earlier, negatives mention "NEGATIVE".
            return "MATCH" if "POSITIVE" in user_msg[:200] else "NO_MATCH"

        if "explainer" in sys_msg or "structured explanation" in sys_msg:
            # LLM2: emit minimum-valid schema referencing two cited features.
            m = re.search(r"i\d+", user_msg)
            cite = m.group(0) if m else "i1"
            return (
                f"S: {{{cite}}}\n"
                f"Mechanisms:\n- The trajectory exhibits the cited pattern. [cite: {cite}]\n"
                f"Aggregation:\n- Cited cue increases support for the predicted class. [cite: {cite}]\n"
                f"Limits:\n- Other cues not used. [cite: {cite}]"
            )

        if "judge" in sys_msg:
            # Judge: emit JSON with a slight bias toward class 1 if "support" appears positive.
            p1 = 0.6 if "increase" in user_msg or "high" in user_msg else 0.4
            return json.dumps({"prob_high": p1, "prob_low": 1 - p1})

        return "(stub: unrecognized role)"


# ---------------------------------------------------------------------------
# DashScope (optional)


@dataclass
class DashScopeClient:
    """Calls Alibaba DashScope's Qwen API. Requires ``DASHSCOPE_API_KEY`` env var."""

    model_id: str = "qwen2.5-7b-instruct"
    name: str = "dashscope"

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        import requests
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if stop:
            payload["stop"] = stop
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def get_default_client(prefer: str = "auto") -> LLMClient:
    """Best-effort default: dashscope if key present, else qwen-local, else stub.

    Pass ``prefer="stub"`` to force the stub (for dry-run notebooks).
    """
    if prefer == "stub":
        return StubClient()
    if prefer in ("dashscope", "auto") and os.environ.get("DASHSCOPE_API_KEY"):
        return DashScopeClient()
    if prefer in ("qwen-local", "auto"):
        return QwenLocalClient()
    return StubClient()

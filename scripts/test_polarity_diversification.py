"""Reload the snapshot saved by run_pipeline.ipynb and re-run only the
LLM2 + Judge stage on inputs #0 and #1, comparing the cherry-picking case
(input #1) before and after polarity-aware K-candidate diversification.

This skips the 15-min LLM1 phase entirely. Total runtime ~1-2 min on
Qwen 2.5-3B (4 candidates per input × 2 inputs × 1 LLM2 + 1 Judge call each).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from the bb_pipeline/ directory or its scripts/ subdir.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from bb_pipeline.blackbox import OutcomeMLP
from bb_pipeline.sae import TopKSAE
from bb_pipeline.state_io import load_pipeline_state
from bb_pipeline.llm import QwenLocalClient
from bb_pipeline.pipeline import explain_with_selection


SNAPSHOT_DIR = HERE.parent / "snapshot" / "run_001"
TEST_INDICES = [163, 1869]   # same as the notebook's first two test inputs


def main():
    print(f"loading snapshot from {SNAPSHOT_DIR} ...")
    state = load_pipeline_state(
        SNAPSHOT_DIR, mlp_class=OutcomeMLP, sae_class=TopKSAE,
    )
    model = state["model"]
    sae = state["sae"]
    X = state["X"]
    Z_va = state["Z_va"]
    va_idx = state["va_idx"]
    scaler = state["scaler"]
    concept_labels = state["concept_labels"]
    polarities = state["polarities"]
    device = next(model.parameters()).device

    print(f"loading Qwen 2.5-3B-Instruct (cached) ...")
    client = QwenLocalClient(model_id="Qwen/Qwen2.5-3B-Instruct")
    _ = client.chat([{"role": "user", "content": "hi"}], max_new_tokens=4)
    print("client warmed up\n")

    # va_idx tells us which row of X each row of Z_va corresponds to.
    # The notebook uses test_pick = rng.choice(len(va), ...) which gave us [163, 1869].
    for k_, vidx in enumerate(TEST_INDICES):
        x = X[va_idx[vidx]]
        z = Z_va[vidx]
        active = [int(j) for j in np.where(z > 0)[0] if j in concept_labels]
        if not active:
            print(f"[skip {k_}] no labeled active features for this input")
            continue
        activ_vals = [float(z[j]) for j in active]
        x_norm = ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)
        with torch.no_grad():
            p_m_raw = model.predict_proba(
                torch.from_numpy(x_norm).to(device).unsqueeze(0)
            ).cpu().numpy().reshape(-1)
        p_m = np.array([float(p_m_raw[0]), float(p_m_raw[1])])

        # Show polarity composition of this input
        low_active = [(f, activ_vals[active.index(f)]) for f in active
                      if f in polarities and polarities[f].direction == "LOW"]
        high_active = [(f, activ_vals[active.index(f)]) for f in active
                       if f in polarities and polarities[f].direction == "HIGH"]
        neut_active = [(f, activ_vals[active.index(f)]) for f in active
                       if f in polarities and polarities[f].direction == "NEUTRAL"]

        print("=" * 70)
        print(f"input #{k_} (val idx {vidx})")
        print(f"  blackbox P(low, high) = ({p_m[0]:.3f}, {p_m[1]:.3f})")
        print(f"  active LOW : {[f for f,_ in low_active]}  (weights: {[round(v,2) for _,v in low_active]})")
        print(f"  active HIGH: {[f for f,_ in high_active]}  (weights: {[round(v,2) for _,v in high_active]})")
        print(f"  active NEUT: {[f for f,_ in neut_active]}")

        res = explain_with_selection(
            client, input_idx=int(vidx),
            p_blackbox=p_m,
            active_features=active, activation_values=activ_vals,
            concept_labels=concept_labels,
            polarities=polarities,
            K=4,
        )
        for c_i, c in enumerate(res.candidates):
            marker = "<-- WIN" if c_i == res.winner_index else ""
            print(
                f"  [{c.strategy:>9s}] cand {c_i}: KL={c.kl:.3f}  |S|={c.sparsity}  "
                f"Len={c.length}  total={c.total_loss:.3f}  "
                f"Judge=({c.judge.prob_low:.2f},{c.judge.prob_high:.2f}) {marker}"
            )
        print(f"  WINNING EXPLANATION (strategy={res.winner.strategy}):")
        print("  " + res.winner.explanation.raw_text.replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()

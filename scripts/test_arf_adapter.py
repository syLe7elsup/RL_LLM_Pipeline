"""Smoke test for arf_adapter: verify we can load LLM_RL artifacts and run
the no-LLM portion of bb_pipeline (evidence collection + polarity) on them.

Pre-requisite: stages 1-5 of LLM_RL must have produced their outputs at
``DEFAULT_ARTIFACT_DIR`` (/Users/abigail/Downloads/llmproject/LLM_RL/data/outputs).

Usage:
    python3 scripts/test_arf_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from bb_pipeline.arf_adapter import load_arf_state
from bb_pipeline.evidence import collect_evidence
from bb_pipeline.feature_polarity import compute_polarities, render_polarity_tag


def main():
    print("Loading LLM_RL artifacts...")
    state = load_arf_state()

    X = state["X"]; y = state["y"]
    tr = state["tr_idx"]; va = state["va_idx"]
    H_tr, H_va = state["H_tr"], state["H_va"]
    Z_tr, Z_va = state["Z_tr"], state["Z_va"]
    names = state["feature_names"]
    a_tr = state["actions_train"]

    print(f"feature names ({len(names)}): {names[:6]} ...")
    print(f"X shape: {X.shape}   y (IMV) balance: {int(y.sum())}/{len(y)}")
    print(f"tr={len(tr)}, va={len(va)}")
    print(f"H_tr {H_tr.shape}, Z_tr {Z_tr.shape}, Z_va {Z_va.shape}")
    print("action distribution train: " +
          ", ".join(f"a={a}:{int((a_tr==a).sum())}" for a in [0, 1, 2]))

    device = next(state["model"].parameters()).device
    x_one = torch.from_numpy(X[tr[0]:tr[0] + 1]).to(device)
    p = state["model"].predict_proba(x_one).cpu().numpy().reshape(-1)
    print(f"\nblackbox shim sample: P(other, IMV) = ({p[0]:.3f}, {p[1]:.3f})")

    print("\nCollecting SAE evidence (latent_dim=128, sparse via L1)...")
    ev = collect_evidence(Z_tr, n_pos=5, n_neg=5, n_pos_val=8, n_neg_val=8, seed=0)
    print(f"features with usable evidence: {len(ev)} / 128")

    print("\nComputing polarities (binary y = action == IMV)...")
    pols = compute_polarities(Z_tr, y[tr], feature_indices=sorted(ev.keys()))
    counts = {"HIGH": 0, "LOW": 0, "NEUTRAL": 0}
    for p in pols.values():
        counts[p.direction] += 1
    print(f"  HIGH (-> IMV):     {counts['HIGH']}")
    print(f"  LOW  (-> non-IMV): {counts['LOW']}")
    print(f"  NEUTRAL:           {counts['NEUTRAL']}")

    print("\nTop 8 features by |correlation|:")
    for fid in sorted(pols, key=lambda i: -abs(pols[i].correlation))[:8]:
        pol = pols[fid]
        print(
            f"  i{fid:>3d}  corr={pol.correlation:+.3f}  "
            f"density={ev[fid].density:.3f}  {render_polarity_tag(pol)}"
        )


if __name__ == "__main__":
    main()

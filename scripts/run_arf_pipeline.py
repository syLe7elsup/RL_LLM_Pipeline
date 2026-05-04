"""End-to-end ARF (DQN-on-clinical-data) explanation pipeline.

Load the artifacts that LLM_RL stages 1-5 produce, then run our optimized
explanation stack on top:

    Stage 6  collect_evidence (binary y = action == IMV)
    Stage 7  LLM1 concept naming (two-pass with gold buffer)
    Stage 7b feature_polarity vs the binary IMV-vs-not label
    Stage 8  K-candidate explanations (free / balanced / low_only / high_only)
             scored against the DQN's softmax(Q) projected to [P_other, P_imv]

A snapshot of the (concepts, polarities, evidence, verifier_acc) is saved to
``arf_snapshot/run_001/`` so subsequent re-runs can reuse the LLM1 phase.

Usage:
    python3 -u scripts/run_arf_pipeline.py --client stub --n_explain 3
    python3 -u scripts/run_arf_pipeline.py --client qwen --n_explain 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from bb_pipeline.arf_adapter import load_arf_state
from bb_pipeline.evidence import collect_evidence, render_evidence_block
from bb_pipeline.feature_polarity import compute_polarities, render_polarity_tag
from bb_pipeline.gold_buffer import GoldBuffer
from bb_pipeline.llm import StubClient, QwenLocalClient
from bb_pipeline.llm1_concept import propose_concept
from bb_pipeline.verifier import EmbeddingVerifier, label_with_refinement
from bb_pipeline.pipeline import explain_with_selection


def print_flush(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client", choices=["stub", "qwen"], default="stub")
    p.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n_explain", type=int, default=10,
                   help="how many test/val patients to explain end-to-end")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.65,
                   help="verifier-accuracy threshold for refinement")
    p.add_argument("--snapshot_dir", default=str(HERE.parent / "arf_snapshot" / "run_001"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_snapshot", action="store_true",
                   help="skip LLM1 + polarity, load from snapshot instead")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print_flush("=== Loading LLM_RL artifacts ===")
    state = load_arf_state()
    X = state["X"]; y = state["y"]
    tr = state["tr_idx"]; va = state["va_idx"]
    Z_tr, Z_va = state["Z_tr"], state["Z_va"]
    names = state["feature_names"]
    X_tr_arr, X_va_arr = X[tr], X[va]
    print_flush(f"  X={X.shape}  y(IMV)={int(y.sum())}/{len(y)}  "
                f"tr/va={len(tr)}/{len(va)}  SAE latents Z_tr={Z_tr.shape}")

    # ---------------- LLM client + verifier ----------------
    print_flush(f"\n=== Loading LLM client (mode={args.client}) ===")
    if args.client == "stub":
        client = StubClient()
    else:
        client = QwenLocalClient(model_id=args.qwen_model)
    verifier = EmbeddingVerifier()
    _ = client.chat([{"role": "user", "content": "hi"}], max_new_tokens=4)
    verifier._ensure_loaded()
    print_flush("  client + verifier warmed up")

    snapshot_dir = Path(args.snapshot_dir)

    # ---------------- LLM1 + polarity (or load) ----------------
    if args.use_snapshot and (snapshot_dir / "concepts.json").exists():
        print_flush(f"\n=== Loading LLM1 snapshot from {snapshot_dir} ===")
        with open(snapshot_dir / "concepts.json") as f:
            concept_labels = {int(k): v for k, v in json.load(f).items()}
        with open(snapshot_dir / "polarities.pkl", "rb") as f:
            polarities = pickle.load(f)
        with open(snapshot_dir / "evidence.pkl", "rb") as f:
            evidence = pickle.load(f)
        with open(snapshot_dir / "verifier_acc.json") as f:
            verifier_acc = {int(k): float(v) for k, v in json.load(f).items()}
        print_flush(f"  loaded {len(concept_labels)} concepts, "
                    f"{sum(1 for a in verifier_acc.values() if a >= args.alpha)} above alpha={args.alpha}")
    else:
        # Fresh LLM1 phase.
        print_flush("\n=== Stage 6: collect_evidence ===")
        evidence = collect_evidence(Z_tr, n_pos=5, n_neg=5,
                                    n_pos_val=8, n_neg_val=8, seed=args.seed)
        print_flush(f"  features with usable evidence: {len(evidence)} / {Z_tr.shape[1]}")

        print_flush("\n=== Stage 7: LLM1 two-pass naming with gold buffer ===")
        concept_labels: dict[int, str] = {}
        verifier_acc: dict[int, float] = {}
        gold = GoldBuffer(max_size=10)
        pass1_failures: list[int] = []

        t0 = time.time()
        for fid in sorted(evidence):
            res, vres = label_with_refinement(
                client, evidence[fid], X_tr_arr, names,
                verifier=verifier,
                alpha=args.alpha, max_rounds=1,
            )
            concept_labels[fid] = res.concept
            verifier_acc[fid] = vres.accuracy
            tag = ""
            if vres.accuracy >= args.alpha:
                gold.push(evidence[fid], res.concept, vres.accuracy)
                tag = "  -> gold"
            else:
                pass1_failures.append(fid)
            print_flush(
                f"  i{fid:>3d}: \"{res.concept}\"  acc={vres.accuracy:.2f}{tag}"
            )
        print_flush(f"\n  pass 1 done in {time.time()-t0:.1f}s; "
                    f"gold buffer={len(gold)}; failures={len(pass1_failures)}")

        # Pass 2 with few-shot from gold buffer
        if pass1_failures and len(gold) > 0:
            print_flush("\n--- Pass 2: re-naming low-acc features using gold buffer ---")
            for fid in pass1_failures:
                new_res, new_vres = label_with_refinement(
                    client, evidence[fid], X_tr_arr, names,
                    verifier=verifier,
                    alpha=args.alpha, max_rounds=1,
                    gold_buffer=gold, gold_k=2, gold_seed=fid,
                )
                old_acc = verifier_acc[fid]
                if new_vres.accuracy >= old_acc:
                    print_flush(f"  i{fid:>3d}: \"{new_res.concept}\"  "
                                f"acc={new_vres.accuracy:.2f}  (was {old_acc:.2f}, kept)")
                    concept_labels[fid] = new_res.concept
                    verifier_acc[fid] = new_vres.accuracy
                else:
                    print_flush(f"  i{fid:>3d}: kept original (Δ negative)")

        print_flush(f"\n  total LLM1 time: {time.time()-t0:.1f}s")
        print_flush(f"  final ≥α: {sum(1 for a in verifier_acc.values() if a >= args.alpha)} / {len(verifier_acc)}")

        print_flush("\n=== Stage 7b: compute polarities ===")
        polarities = compute_polarities(
            Z_tr, y[tr], feature_indices=sorted(concept_labels.keys())
        )
        ct = {"HIGH": 0, "LOW": 0, "NEUTRAL": 0}
        for pol in polarities.values():
            ct[pol.direction] += 1
        print_flush(f"  HIGH(-> IMV)={ct['HIGH']}  LOW(-> non-IMV)={ct['LOW']}  NEUTRAL={ct['NEUTRAL']}")

        # Save snapshot
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with open(snapshot_dir / "concepts.json", "w") as f:
            json.dump({str(k): v for k, v in concept_labels.items()}, f, indent=2)
        with open(snapshot_dir / "polarities.pkl", "wb") as f:
            pickle.dump(polarities, f)
        with open(snapshot_dir / "evidence.pkl", "wb") as f:
            pickle.dump(evidence, f)
        with open(snapshot_dir / "verifier_acc.json", "w") as f:
            json.dump({str(k): float(v) for k, v in verifier_acc.items()}, f, indent=2)
        print_flush(f"  snapshot saved to {snapshot_dir}")

    # ---------------- Stage 8: K-candidate explanations ----------------
    print_flush(f"\n=== Stage 8: explanations on N={args.n_explain} val patients ===")
    test_pick = rng.choice(len(va), size=args.n_explain, replace=False)
    device = next(state["model"].parameters()).device
    results = []

    for k_, vidx in enumerate(test_pick):
        x = X_va_arr[vidx]
        z = Z_va[vidx]
        active = [int(j) for j in np.where(z > 0)[0] if j in concept_labels]
        if not active:
            print_flush(f"\n  [skip {k_}] no labeled active features for vidx={vidx}")
            continue
        activ_vals = [float(z[j]) for j in active]
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(device).unsqueeze(0)
            p_raw = state["model"].predict_proba(x_t).cpu().numpy().reshape(-1)
        p_m = np.array([float(p_raw[0]), float(p_raw[1])])

        bb_argmax = int(np.argmax(p_m))
        print_flush(f"\n  ── input #{k_} (vidx={vidx}) ──")
        print_flush(f"    blackbox P(other,IMV)=({p_m[0]:.3f},{p_m[1]:.3f})  "
                    f"argmax={'IMV' if bb_argmax==1 else 'other'}")
        print_flush(f"    active feats: {len(active)}")
        # Polarity composition
        low_ct = sum(1 for f in active if f in polarities and polarities[f].direction == "LOW")
        high_ct = sum(1 for f in active if f in polarities and polarities[f].direction == "HIGH")
        print_flush(f"    polarity split: {low_ct} LOW, {high_ct} HIGH")

        res = explain_with_selection(
            client, input_idx=int(vidx),
            p_blackbox=p_m,
            active_features=active, activation_values=activ_vals,
            concept_labels=concept_labels, polarities=polarities,
            K=args.K,
        )
        winner = res.winner
        agree = bb_argmax == int(np.argmax(winner.judge.probs))
        for c_i, c in enumerate(res.candidates):
            mark = "<- WIN" if c_i == res.winner_index else ""
            print_flush(f"    [{c.strategy:>9s}] KL={c.kl:.3f}  "
                        f"Judge=({c.judge.prob_low:.2f},{c.judge.prob_high:.2f}) {mark}")
        print_flush(f"    argmax_agree={agree}")
        results.append({
            "vidx": int(vidx),
            "p_blackbox": p_m.tolist(),
            "p_judge": [float(winner.judge.prob_low), float(winner.judge.prob_high)],
            "kl": float(winner.kl),
            "argmax_agree": bool(agree),
            "winning_strategy": winner.strategy,
            "n_active": len(active),
            "low_active": low_ct,
            "high_active": high_ct,
        })

    # ---------------- Final summary ----------------
    if results:
        kls = [r["kl"] for r in results]
        agreed = sum(r["argmax_agree"] for r in results)
        from collections import Counter
        strat = Counter(r["winning_strategy"] for r in results)
        print_flush("\n" + "=" * 60)
        print_flush(f"ARF eval summary  (N={len(results)})")
        print_flush("=" * 60)
        print_flush(f"  argmax agreement: {agreed}/{len(results)} = {agreed/len(results):.3f}")
        print_flush(f"  KL mean={np.mean(kls):.3f}  median={np.median(kls):.3f}  max={np.max(kls):.3f}")
        print_flush(f"  winning strategies: {dict(strat)}")
        out_path = snapshot_dir / f"explanations_n{args.n_explain}.json"
        out_path.write_text(json.dumps({"summary": {
            "N": len(results),
            "argmax_agree_rate": agreed / len(results),
            "kl_mean": float(np.mean(kls)),
            "kl_median": float(np.median(kls)),
            "kl_max": float(np.max(kls)),
            "winning_strategies": dict(strat),
        }, "per_input": results}, indent=2))
        print_flush(f"  results saved to {out_path}")


if __name__ == "__main__":
    main()

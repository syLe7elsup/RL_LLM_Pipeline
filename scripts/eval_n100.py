"""Evaluate the explanation pipeline on N test inputs and report real metrics.

Loads the snapshot saved by run_pipeline.ipynb (so the expensive LLM1 phase
doesn't have to re-run), then runs LLM2 + Judge + K-candidate selection on
N validation inputs and reports:

    - argmax agreement rate: P(argmax(Judge) == argmax(blackbox))
    - KL distribution: mean / median / p25 / p75 / p95 / max
    - winning strategy distribution (free / balanced / low_only / high_only)
    - per-input results saved to scripts/eval_n100_results.json for analysis

Default N=100. Estimated runtime on Qwen 2.5-3B Instruct (MPS): ~80 min.

Usage:
    python3 scripts/eval_n100.py                  # default N=100
    python3 scripts/eval_n100.py --N 30           # quicker
    python3 scripts/eval_n100.py --K 2            # fewer candidates per input
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from bb_pipeline.blackbox import OutcomeMLP
from bb_pipeline.sae import TopKSAE
from bb_pipeline.state_io import load_pipeline_state
from bb_pipeline.llm import QwenLocalClient
from bb_pipeline.pipeline import explain_with_selection


def print_flush(*args, **kwargs):
    """print() with stdout flush — for visibility under stdout buffering."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=100, help="number of validation inputs to explain")
    p.add_argument("--K", type=int, default=4, help="number of candidates per input")
    p.add_argument("--snapshot", default=str(HERE.parent / "snapshot" / "run_001"))
    p.add_argument("--out", default=str(HERE / "eval_n100_results.json"))
    p.add_argument("--seed", type=int, default=42, help="seed for picking val indices")
    args = p.parse_args()

    print_flush(f"loading snapshot from {args.snapshot} ...")
    state = load_pipeline_state(args.snapshot, mlp_class=OutcomeMLP, sae_class=TopKSAE)
    model = state["model"]
    sae = state["sae"]
    X = state["X"]
    Z_va = state["Z_va"]
    va_idx = state["va_idx"]
    scaler = state["scaler"]
    concept_labels = state["concept_labels"]
    polarities = state["polarities"]
    device = next(model.parameters()).device

    print_flush(f"loading Qwen 2.5-3B-Instruct (cached) ...")
    client = QwenLocalClient(model_id="Qwen/Qwen2.5-3B-Instruct")
    _ = client.chat([{"role": "user", "content": "hi"}], max_new_tokens=4)
    print_flush("client warmed up\n")

    rng = np.random.default_rng(args.seed)
    val_pool = rng.choice(len(va_idx), size=args.N * 2, replace=False)  # over-sample, filter

    per_input = []
    skipped = 0
    t_start = time.time()

    for k_, vidx_local in enumerate(val_pool):
        if len(per_input) >= args.N:
            break
        vidx = int(vidx_local)
        x = X[va_idx[vidx]]
        z = Z_va[vidx]
        active = [int(j) for j in np.where(z > 0)[0] if j in concept_labels]
        if not active:
            skipped += 1
            continue
        activ_vals = [float(z[j]) for j in active]
        x_norm = ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)
        with torch.no_grad():
            p_raw = model.predict_proba(
                torch.from_numpy(x_norm).to(device).unsqueeze(0)
            ).cpu().numpy().reshape(-1)
        p_m = np.array([float(p_raw[0]), float(p_raw[1])])

        try:
            res = explain_with_selection(
                client, input_idx=vidx,
                p_blackbox=p_m,
                active_features=active, activation_values=activ_vals,
                concept_labels=concept_labels,
                polarities=polarities,
                K=args.K,
            )
        except Exception as e:
            print_flush(f"  [error on vidx={vidx}] {type(e).__name__}: {e}")
            skipped += 1
            continue

        winner = res.winner
        bb_argmax = int(np.argmax(p_m))
        judge_argmax = int(np.argmax(winner.judge.probs))
        agree = bb_argmax == judge_argmax

        # Polarity composition of this input
        low_ct = sum(1 for f in active if f in polarities and polarities[f].direction == "LOW")
        high_ct = sum(1 for f in active if f in polarities and polarities[f].direction == "HIGH")
        is_mixed = low_ct > 0 and high_ct > 0

        per_input.append({
            "vidx": vidx,
            "p_blackbox_low": float(p_m[0]),
            "p_blackbox_high": float(p_m[1]),
            "p_judge_low": float(winner.judge.prob_low),
            "p_judge_high": float(winner.judge.prob_high),
            "kl": float(winner.kl),
            "argmax_agree": bool(agree),
            "winning_strategy": winner.strategy,
            "n_active": len(active),
            "n_low_active": low_ct,
            "n_high_active": high_ct,
            "is_mixed_polarity": bool(is_mixed),
            "sparsity": int(winner.sparsity),
            "length": int(winner.length),
            "all_candidate_kls": [float(c.kl) for c in res.candidates],
            "all_candidate_strategies": [c.strategy for c in res.candidates],
        })

        if (len(per_input)) % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / len(per_input) * (args.N - len(per_input))
            kls_so_far = [r["kl"] for r in per_input]
            agree_so_far = sum(r["argmax_agree"] for r in per_input)
            print_flush(
                f"  [{len(per_input):>3d}/{args.N}] elapsed={elapsed:>5.0f}s  ETA={eta:>5.0f}s  "
                f"argmax_agree={agree_so_far}/{len(per_input)}  "
                f"mean_KL={np.mean(kls_so_far):.3f}  median_KL={np.median(kls_so_far):.3f}"
            )
            # Persist a partial dump in case we have to kill mid-run.
            partial_path = str(args.out) + ".partial"
            Path(partial_path).write_text(json.dumps({
                "in_progress": True, "completed": len(per_input),
                "per_input": per_input,
            }, indent=2))

    # ---------- summary ----------
    if not per_input:
        print_flush("no inputs processed.")
        return

    kls = [r["kl"] for r in per_input]
    agrees = [r["argmax_agree"] for r in per_input]
    mixed = [r for r in per_input if r["is_mixed_polarity"]]
    pure = [r for r in per_input if not r["is_mixed_polarity"]]

    summary = {
        "N_processed": len(per_input),
        "N_skipped": skipped,
        "argmax_agree_rate": sum(agrees) / len(agrees),
        "kl_mean": float(np.mean(kls)),
        "kl_median": float(np.median(kls)),
        "kl_p25": percentile(kls, 25),
        "kl_p75": percentile(kls, 75),
        "kl_p95": percentile(kls, 95),
        "kl_max": float(np.max(kls)),
        "winning_strategy_counts": {},
        "by_polarity": {},
    }

    from collections import Counter
    strat_counter = Counter(r["winning_strategy"] for r in per_input)
    summary["winning_strategy_counts"] = dict(strat_counter)

    for label, group in [("mixed_polarity", mixed), ("single_polarity", pure)]:
        if not group:
            continue
        summary["by_polarity"][label] = {
            "n": len(group),
            "argmax_agree_rate": sum(r["argmax_agree"] for r in group) / len(group),
            "kl_mean": float(np.mean([r["kl"] for r in group])),
            "kl_median": float(np.median([r["kl"] for r in group])),
        }

    print_flush("\n" + "=" * 60)
    print_flush(f"Eval summary  (N={summary['N_processed']}, skipped={summary['N_skipped']}, K={args.K})")
    print_flush("=" * 60)
    print_flush(f"  argmax agreement rate: {summary['argmax_agree_rate']:.3f}  ({sum(agrees)}/{len(agrees)})")
    print_flush(f"  KL  mean={summary['kl_mean']:.3f}  median={summary['kl_median']:.3f}")
    print_flush(f"      p25={summary['kl_p25']:.3f}  p75={summary['kl_p75']:.3f}  p95={summary['kl_p95']:.3f}  max={summary['kl_max']:.3f}")
    print_flush(f"  winning strategies: {summary['winning_strategy_counts']}")
    if summary["by_polarity"]:
        print_flush(f"  by polarity:")
        for label, stats in summary["by_polarity"].items():
            print_flush(f"    {label:>16s}: n={stats['n']:>3d}  agree={stats['argmax_agree_rate']:.3f}  KL_mean={stats['kl_mean']:.3f}  KL_median={stats['kl_median']:.3f}")

    out = {"summary": summary, "per_input": per_input}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print_flush(f"\nfull results saved to {args.out}")


if __name__ == "__main__":
    main()

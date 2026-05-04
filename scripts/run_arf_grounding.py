"""Run the ground-truth concept grounding analysis on ARF.

Loads ground-truth concepts (LLM_RL stage 1 output) and the SAE latents
+ patient_id/time_step indices (LLM_RL stage 5 + arf_adapter), aligns
them, computes the |Pearson r| matrix between every alive SAE feature and
every ground-truth concept, and prints a report. Optionally cross-
references with our LLM1 concept names so we can ask: "the SAE feature
that LLM1 named X — does it actually correspond to ground-truth concept Y?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from bb_pipeline.arf_adapter import (
    DEFAULT_ARTIFACT_DIR, load_arf_state,
)
from bb_pipeline.concept_grounding import (
    GROUND_TRUTH_COLS,
    align_concepts_to_latents,
    best_match_per_concept,
    best_match_per_feature,
    compute_grounding_matrix,
    load_ground_truth_concepts,
    report_grounding,
)


DEFAULT_CONCEPT_CSV = (
    Path("/Users/abigail/Downloads/llmproject/LLM_RL/data/raw")
    / "toy_arf_ground_truth_concepts.csv"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concept_csv", default=str(DEFAULT_CONCEPT_CSV),
                   help="path to LLM_RL stage-1 toy_arf_ground_truth_concepts.csv")
    p.add_argument("--snapshot", default=str(HERE.parent / "arf_snapshot" / "run_001"),
                   help="(optional) bb_pipeline snapshot dir for LLM1 concept names")
    p.add_argument("--split", choices=["train", "val", "both"], default="both",
                   help="which SAE latents to use for the correlation")
    p.add_argument("--out", default=str(HERE.parent / "arf_snapshot" / "run_001" / "grounding.json"),
                   help="where to save the per-feature best-match JSON")
    args = p.parse_args()

    print("Loading ARF state (DQN + SAE artifacts) ...")
    state = load_arf_state()

    print(f"Loading ground-truth concepts from {args.concept_csv}")
    concept_df = load_ground_truth_concepts(args.concept_csv)
    print(f"  concept_df: {concept_df.shape}")

    # Load patient_id / time_step lists (saved by stage 4 alongside hidden states)
    artifact_dir = DEFAULT_ARTIFACT_DIR
    train_h_dict = torch.load(artifact_dir / "train_hidden_states.pt", weights_only=False)
    val_h_dict   = torch.load(artifact_dir / "val_hidden_states.pt",   weights_only=False)

    if args.split == "train":
        Z = state["Z_tr"]
        pids = train_h_dict["patient_ids"]
        tss  = train_h_dict["time_steps"]
    elif args.split == "val":
        Z = state["Z_va"]
        pids = val_h_dict["patient_ids"]
        tss  = val_h_dict["time_steps"]
    else:
        Z = np.concatenate([state["Z_tr"], state["Z_va"]], axis=0)
        pids = list(train_h_dict["patient_ids"]) + list(val_h_dict["patient_ids"])
        tss  = list(train_h_dict["time_steps"])  + list(val_h_dict["time_steps"])
    print(f"  using {args.split} split: Z={Z.shape}")

    print("Aligning concepts to latents by (patient_id, time_step) ...")
    C = align_concepts_to_latents(concept_df, pids, tss)
    print(f"  aligned matrix C: {C.shape}")

    feature_indices = [i for i in range(Z.shape[1]) if (Z[:, i] > 0).sum() > 0]
    print(f"  alive features: {len(feature_indices)}")

    print("\nComputing |Pearson r| grounding matrix ...")
    grounding = compute_grounding_matrix(Z, C, feature_indices, use_abs=True)

    # Optional: load LLM1 concept names for cross-referencing
    concept_labels = None
    snap = Path(args.snapshot)
    cl_path = snap / "concepts.json"
    if cl_path.exists():
        with cl_path.open() as f:
            concept_labels = {int(k): v for k, v in json.load(f).items()}
        print(f"  cross-referencing with {len(concept_labels)} LLM1 concept names from {cl_path}")

    print()
    print(report_grounding(grounding, feature_indices, concept_labels=concept_labels))

    # Save machine-readable per-feature best matches.
    feat_match = best_match_per_feature(grounding, feature_indices)
    out_data = {
        "split": args.split,
        "n_aligned_rows": int((~np.isnan(C).any(axis=1)).sum()),
        "feature_best_match": {
            int(fid): {
                "concept": m.concept,
                "abs_correlation": float(m.correlation),
                "llm1_label": (concept_labels.get(fid) if concept_labels else None),
            }
            for fid, m in feat_match.items()
        },
        "concept_best_feature": {
            name: {
                "feature_idx": int(m.feature_idx),
                "abs_correlation": float(m.correlation),
                "llm1_label": (concept_labels.get(m.feature_idx) if concept_labels else None),
            }
            for name, m in best_match_per_concept(grounding, feature_indices).items()
        },
    }
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"\nFull grounding saved to {args.out}")


if __name__ == "__main__":
    main()

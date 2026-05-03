# RL_LLM_Pipeline

Reproduction of the SAE-based blackbox-explanation pipeline from *Explain_BB.pdf*,
adapted to the POLAR simulated 3-stage Dynamic Treatment Regime data.

## What this is

A pipeline that explains a small classifier's predictions on patient
treatment trajectories by:

1. Training a small MLP **blackbox** to predict outcome from a partial
   trajectory (`s_1, a_1, s_2, a_2, s_3, a_3` → high/low outcome).
2. Training a top-k **Sparse Autoencoder** on the blackbox's hidden layer.
3. Asking an **LLM** to name each SAE feature using contrastive evidence
   sets (LLM1).
4. Verifying each name on held-out evidence with a fast embedding
   verifier (`sentence-transformers/all-MiniLM-L6-v2`) calibrated per-concept.
5. For each test input, generating K candidate **structured explanations**
   that cite only the named concepts (LLM2), and picking the one whose
   **Judge** prediction (from explanation alone, no input access) best
   matches the blackbox.

## Why this setup

POLAR simulated data gives us:

- a known data-generating process (so we can sanity-check what the SAE
  decomposes vs the actual underlying structure),
- low cost (no DUA needed, runs end-to-end in ~20 min on a single GPU),
- a non-trivial classification task once `s_4` is excluded from input
  (the reward depends on `s_4`, so leaving it in collapses the task to
  closed-form formula inversion).

## Data

POLAR is *Pessimistic Model-based Policy Learning Algorithm for DTRs*
(<https://github.com/Papillon-Xiang/POLAR-blind>). The data generator in
`polar_data.py` is copied from that repo with attribution; the rest is new.

Each trajectory: `(s_1, a_1, s_2, a_2, s_3, a_3, s_4)` with `s_k ∈ [0,1]^2`
and `a_k ∈ {0,1}`. Reward is computed only at stage 3 from `s_3` and `s_4`.

## Module layout

| file | responsibility |
| --- | --- |
| `polar_data.py` | POLAR trajectory generator (copied from POLAR-blind) |
| `features.py` | 28-dim derived feature engineering (deltas, treatment patterns, summaries) |
| `blackbox.py` | MLP `28 → 64 → 64 → 2`, dropout / weight decay / early stopping |
| `sae.py` | Top-k SAE with Gao 2024 dead-feature auxiliary loss |
| `evidence.py` | Per-feature `(E^+, E^-)` collection + rendering |
| `feature_polarity.py` | Per-feature directional polarity (HIGH / LOW) wrt the label |
| `llm.py` | Pluggable LLM client: `QwenLocalClient`, `StubClient`, `DashScopeClient` |
| `llm1_concept.py` | Concept proposal + refinement prompts (gold-buffer-aware) |
| `verifier.py` | `LLMVerifier` and `EmbeddingVerifier` (per-concept threshold calibration) |
| `gold_buffer.py` | FIFO few-shot buffer of high-acc concepts |
| `llm2_explainer.py` | Constrained structured explanation with citation discipline + polarity tags |
| `judge.py` | Predicts `P(class)` from the explanation alone, computes KL vs blackbox |
| `pipeline.py` | K-candidate explanation generation + selection |
| `state_io.py` | Snapshot / restore the expensive intermediate state |
| `run_pipeline.ipynb` | End-to-end notebook (default uses `StubClient` for instant dry-run) |

## Quick start

### Local

```bash
pip install -r requirements.txt
jupyter notebook run_pipeline.ipynb
```

The notebook auto-picks `Qwen2.5-3B-Instruct` if you have a GPU (CUDA or
Apple MPS). To dry-run the plumbing without an LLM, replace the
`QwenLocalClient(...)` line in cell 14 with `client = StubClient()`.

### Google Colab

1. Open Colab → File → Upload `run_pipeline.ipynb`
2. Runtime → Change runtime type → T4 GPU (free) or higher
3. Run all

The first cell auto-detects Colab, installs deps, clones this repo, and
`cd`s into it. Subsequent cells then run unchanged — and on a Colab T4
the model auto-upgrades to `Qwen2.5-7B-Instruct` (~14GB fp16). Snapshots
land in your Google Drive at `MyDrive/bb_pipeline_snapshots/run_001/`
so they survive runtime disconnects.

## Status

Working end-to-end with Qwen 2.5-3B-Instruct + EmbeddingVerifier + Gold
Buffer + polarity-aware LLM2. See the snapshot helper (`state_io.py`) for
fast iteration on the explanation side without re-running the ~17min
naming phase.

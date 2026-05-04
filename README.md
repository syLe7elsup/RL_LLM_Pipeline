# RL_LLM_Pipeline

SAE-based blackbox-explanation pipeline (after *Explain_BB.pdf*) running on
two data backends:

- **POLAR** — abstract 3-stage Dynamic Treatment Regime simulation, MLP
  classifier (binary outcome). Self-contained.
- **ARF** — toy clinical Acute Respiratory Failure simulator with a DQN
  policy (3 actions). Stages 1-5 are produced by the sister
  [`LLM_RL`](https://github.com/Papillon-Xiang/LLM_RL) repo; this repo
  adapts those artifacts into our explanation stack.

Same blackbox-explanation logic in both: SAE on the backbone's hidden
layer → LLM names features → embedding verifier → constrained LLM
explanation cited only from the dictionary → Judge predicts from the
explanation alone → KL(blackbox ∥ Judge) + sparsity / length penalties.

## Headline results

| dataset | backbone   | N  | argmax agree | KL median | KL max  | notes |
| ------- | ---------- | -:| ------------:| ---------:| -------:| --- |
| POLAR   | MLP (binary) | 50 | **0.980**    | **0.024** | 0.165   | mature; cherry-picking fix landed |
| ARF     | DQN (binary IMV vs other) — marginal polarity | 10 | 0.60 | 0.288 | 2.084 | 3 critical-care cases fail |
| ARF     | DQN — **per-input attribution** | 10 | **0.80** | **0.115** | **0.681** | fixes 3/3 critical-care cases |

Full per-experiment writeups: `RESULTS.md` (POLAR), `RESULTS_arf.md` (ARF).

## Pipeline at a glance

```
                        ┌──────────────────────────────────────────────┐
   Trajectory  ─►  Backbone  ──hidden──►  SAE  ──sparse latent z──►   Pipeline
                  (MLP / DQN)                                            │
                                          ▲                              │
                                          │                              │
                                evidence sets E_i⁺ / E_i⁻                │
                                          │                              │
                                LLM1 (Qwen 2.5-3B + Embedding verifier)  │
                                          │                              │
                            concept dict {i → "high SpO2/FiO2"}          │
                                          │                              │
                            polarity / per-input attribution             │
                                          │                              │
                                    LLM2 (K candidates)                  │
                                          │                              │
                                Judge — predicts from explanation only  ◄┘
                                          │
                              KL(P_M ∥ P_Judge) + λ·|S| + λ'·Len(E)
```

## Datasets

### POLAR (built-in)

`polar_data.py` is a copy of the data generator from
[Papillon-Xiang/POLAR-blind](https://github.com/Papillon-Xiang/POLAR-blind/blob/main/simulation/POLAR.py)
(with attribution). Trajectory layout:

```
(s_1, a_1, s_2, a_2, s_3, a_3, s_4)     s_k ∈ [0,1]²,  a_k ∈ {0,1}
reward = (cos(-π·s3_1) + 2cos(π·s3_2) + s4_1 + 2·s4_2 − 1.37) × 3.8
```

Binary classification target: `y = 1[reward > median]`. We deliberately
exclude `s_4` from the input features so the task isn't a closed-form
formula inversion.

### ARF (via LLM_RL artifacts)

The toy ARF simulator is embedded as `arf_data.py` (also produces 6
ground-truth latent concepts). The backbone (DQN) and SAE are trained
by [`Papillon-Xiang/LLM_RL`](https://github.com/Papillon-Xiang/LLM_RL)
stages 1-5; we load those artifacts via `arf_adapter.py`.

State: 27 clinical features (SpO2, FiO2, lactate, MAP, etc.); 3 actions
(`standard_oxygen` / `hfnc_or_niv` / `imv`). Action 1 (HFNC) is <1%
of training data, so we collapse to a binary task `y = (action == IMV)`
to reuse the binary explanation stack unchanged.

To regenerate the ARF artifacts (~3 min CPU):

```bash
# from inside a clone of LLM_RL
python3 run_pipeline.py --start-stage 3 --end-stage 5
```

Then point `arf_adapter.DEFAULT_ARTIFACT_DIR` at `LLM_RL/data/outputs`.

## Module layout

| file                       | role                                                                  |
| -------------------------- | --------------------------------------------------------------------- |
| `polar_data.py`            | POLAR trajectory generator (copied with attribution)                  |
| `features.py`              | 28-dim derived features for POLAR (deltas, treatment patterns)        |
| `blackbox.py`              | MLP `28 → 64 → 64 → 2` for POLAR                                       |
| `arf_data.py`              | toy ARF simulator (copied with attribution; 6 ground-truth concepts) |
| `arf_adapter.py`           | loader for LLM_RL stage-1..5 artifacts; binary IMV-vs-other shim     |
| `sae.py`                   | top-k SAE with Gao 2024 dead-feature aux loss                         |
| `evidence.py`              | per-feature `(E^+, E^-)` collection + textual rendering              |
| `feature_polarity.py`      | marginal polarity (HIGH / LOW / NEUTRAL) per SAE feature             |
| `per_input_attribution.py` | gradient × activation; replaces marginal polarity per-input          |
| `llm.py`                   | pluggable LLM client (`QwenLocalClient` / `StubClient` / DashScope)  |
| `llm1_concept.py`          | concept naming + refinement prompts (gold-buffer-aware)              |
| `verifier.py`              | `LLMVerifier` + `EmbeddingVerifier` (per-concept threshold)          |
| `gold_buffer.py`           | FIFO few-shot buffer of high-acc concepts                             |
| `llm2_explainer.py`        | constrained structured explanation with citation + polarity tags      |
| `judge.py`                 | predicts P(class) from explanation alone; KL vs blackbox             |
| `pipeline.py`              | K-candidate generation (`free` / `balanced` / `low_only` / `high_only`) + winner selection |
| `state_io.py`              | snapshot / restore the expensive intermediate state (POLAR)           |
| `run_pipeline.ipynb`       | POLAR end-to-end notebook (auto-detects Colab vs local)              |
| `scripts/`                 | reproduction scripts — see below                                      |

## Scripts

| script | purpose |
| --- | --- |
| `scripts/extract_results.py`  | pull printed cell output from an executed notebook into `RESULTS.md` |
| `scripts/eval_n100.py`        | POLAR snapshot eval — runs LLM2 + Judge + K-candidate on N val patients |
| `scripts/test_arf_adapter.py` | sanity-check ARF artifact loading + evidence + polarity (no LLM) |
| `scripts/run_arf_pipeline.py` | ARF end-to-end (LLM1 + verifier + LLM2 + Judge); supports `--use_attribution` and `--use_snapshot` |
| `scripts/test_polarity_diversification.py` | snapshot-based test of K-candidate polarity diversification |

## Key design choices

1. **Binary task, three actions in ARF** — collapsing 3-way to binary IMV
   lets the rest of the stack (Judge, polarity, K-candidate strategies)
   work unchanged. HFNC <1% of data so signal loss is small.
2. **Embedding verifier with per-concept threshold calibration** — sweep
   the cosine threshold on each concept's train evidence, cache the best.
   100× faster than LLM verifier and more reliable.
3. **Gold buffer two-pass** — pass 1 names every alive feature without
   buffer; pass 2 re-names features below α with worked examples from the
   pass-1 winners as few-shot context.
4. **Polarity-aware K-candidate diversification** — when active features
   include both LOW- and HIGH-pushing concepts, prescribe candidate S
   subsets explicitly (free / balanced / low_only / high_only) so the
   Judge picks the explanation whose direction best matches the blackbox.
   Solved the cherry-picking failure mode on POLAR.
5. **Per-input attribution overrides marginal polarity** (ARF only) —
   gradient × activation on the IMV logit through the SAE decoder gives
   per-input direction. Fixes the non-monotonic-combination failure mode
   on critical-care patients where marginal polarity is misleading.
6. **Snapshot save/load** — the LLM1 phase is the expensive part (~17
   min on Qwen 3B / MPS). Snapshot the result once; iterate on
   LLM2 / Judge / loss / polarity in seconds.

## Quick start

### Local (POLAR)

```bash
pip install -r requirements.txt
jupyter notebook run_pipeline.ipynb
```

The notebook auto-picks `Qwen2.5-3B-Instruct` if you have a GPU (CUDA or
Apple MPS). Replace the `QwenLocalClient(...)` line in cell 14 with
`client = StubClient()` to dry-run the plumbing instantly.

### Google Colab (POLAR)

1. File → Upload `run_pipeline.ipynb`
2. Runtime → Change runtime type → T4 GPU (free)
3. Run all

The first cell auto-detects Colab, installs deps, clones this repo, and
chdirs into it. On Colab the model auto-upgrades to `Qwen2.5-7B-Instruct`
(~14 GB fp16). Snapshots land at
`MyDrive/bb_pipeline_snapshots/run_001/` so they survive runtime
disconnects.

### Local (ARF)

Pre-requisite: clone & run [`LLM_RL`](https://github.com/Papillon-Xiang/LLM_RL)
stages 1-5 to produce DQN + SAE artifacts. Then:

```bash
# end-to-end (one-time LLM1 ~25 min, then explanations ~12 min/input):
python3 -u scripts/run_arf_pipeline.py --client qwen --n_explain 10

# fast iteration on the explanation side after the first run:
python3 -u scripts/run_arf_pipeline.py --client qwen --use_snapshot \
    --use_attribution --n_explain 10
```

`--use_snapshot` skips the LLM1 phase. `--use_attribution` swaps marginal
polarity for per-input attribution (recommended on ARF).

## Status

| capability                                              | POLAR | ARF |
| ------------------------------------------------------- | :---: | :---: |
| End-to-end run                                          | ✅    | ✅    |
| Embedding verifier + per-concept threshold              | ✅    | ✅    |
| Gold buffer few-shot, two-pass naming                   | ✅    | ✅    |
| Polarity-aware K-candidate diversification              | ✅    | ✅    |
| Per-input attribution (replaces marginal polarity)      | n/a   | ✅    |
| Snapshot save/load                                      | ✅    | ✅    |
| Notebook (Colab-portable)                               | ✅    | —     |
| Standalone runner script                                | —     | ✅    |
| Stage-10 ground-truth concept grounding                 | n/a   | ⏳    |
| 3-class generalization (preserve HFNC)                  | n/a   | ⏳    |

## Open work

- **Sum-attribution variant** to fix the one regression (vidx=412) where
  sign-counting with many small +attr features was misled by a few large
  −attr ones.
- **Stage-10 grounding** — quantitative comparison of the 38 alive ARF
  SAE features against the 6 ground-truth latent concepts emitted by the
  simulator. Pure compute, no LLM. Highest-value "this works on real
  structure" demo.
- **3-class Judge** — keep HFNC. The single-polarity ARF failures might
  be an HFNC-vs-IMV boundary that our binary collapse hides.
- **Concept DSL** — let LLM1 emit executable predicates
  (`max_ind2 > 0.7 AND treatment_count >= 2`) so the verifier can
  evaluate them directly. Expected to push verifier accuracy past 95 %.

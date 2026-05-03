# Run results

Generated from `run_pipeline.ipynb`. Captures the printed output from a single executed run, trimmed to the cells most useful as a reference.

To reproduce: open the notebook, switch the LLM client cell from `StubClient` to `QwenLocalClient(...)`, run all.

## 1. Config + device

```
device = mps
{
  "n": "10000",
  "p": "0.95",
  "train_frac": "0.8",
  "mlp_hidden": "64",
  "mlp_dropout": "0.1",
  "mlp_weight_decay": "0.001",
  "mlp_epochs": "300",
  "sae_latent": "32",
  "sae_k": "8",
  "sae_epochs": "150",
  "n_pos_train": "5",
  "n_neg_train": "5",
  "n_pos_val": "8",
  "n_neg_val": "8",
  "refinement_alpha": "0.7",
  "refinement_max_rounds": "1",
  "explain_K": "3",
  "n_explain_inputs": "2",
  "seed": "0"
}
```

## 2. POLAR data generation

```
trajectories: (10000, 11) in 0.1s
first row: [0.55 0.72 1.   0.47 0.81 1.   0.33 0.31 0.   0.55 0.69]
```

## 3. Derived features + outcome labels

```
X: (10000, 28)  (28 named features)
y: high/low balance = 5000 / 10000
reward range: [-11.49, 14.25], threshold = 0.61
feature names:
   0  s1_ind1
   1  s1_ind2
   2  s2_ind1
   3  s2_ind2
   4  s3_ind1
   5  s3_ind2
   6  a1
   7  a2
   8  a3
   9  delta1_ind1
  10  delta1_ind2
  11  delta2_ind1
  12  delta2_ind2
  13  treatment_count
  14  all_treated
  15  never_treated
  16  switch_count
  17  early_treated
  18  mean_ind1
  19  mean_ind2
  20  max_ind1
  21  max_ind2
  22  min_ind1
  23  min_ind2
  24  s1_x_s3_ind1
  25  s1_x_s3_ind2
  26  ind1_range
  27  ind2_range
```

## 4. MLP blackbox training

```
MLP trained in 2.0s
  train_acc=0.907  val_acc=0.906  best_epoch=14
  overfit_warning=False
```

## 5. SAE training

```
hidden reps: train=(8000, 64), val=(2000, 64), mean nonzero frac=0.685
SAE trained in 45.7s
  explained_variance=0.987
  alive features (density >= 0.5%) = 18 / 32
  per-feature density distribution:
    i 0  0.475  ██████████████
    i 1  0.437  █████████████
    i 2  0.479  ██████████████
    i 3  0.419  ████████████
    i 4  0.531  ███████████████
    i 5  0.624  ██████████████████
    i 6  0.394  ███████████
    i 7  0.001   (DEAD)
    i 8  0.463  █████████████
    i 9  0.002   (DEAD)
    i10  0.000   (DEAD)
    i11  0.000   (DEAD)
    i12  0.000   (DEAD)
    i13  0.502  ███████████████
    i14  0.000   (DEAD)
    i15  0.496  ██████████████
    i16  0.000   (DEAD)
    i17  0.000   (DEAD)
    i18  0.000   (DEAD)
    i19  1.000  ██████████████████████████████
    i20  0.000   (DEAD)
    i21  0.000   (DEAD)
    i22  0.639  ███████████████████
    i23  0.000   (DEAD)
    i24  0.257  ███████
    i25  0.006  
    i26  0.338  ██████████
    i27  0.019  
    i28  0.001   (DEAD)
    i29  0.486  ██████████████
    i30  0.430  ████████████
    i31  0.000   (DEAD)
```

## 6. Evidence collection

```
features with usable evidence: 17 / 32

--- feature 0 (density=0.482) ---
### POSITIVE (feature strongly active)
Example 1:
  - treatment_count=+1.00
  - a3=+1.00
  - switch_count=+1.00
  - max_ind2=+0.87
  - s3_ind2=+0.87
  - s3_ind1=+0.86
  - max_ind1=+0.86
  - s1_ind2=+0.77
Example 2:
  - treatment_count=+2.00
  - switch_count=+2.00
  - a1=+1.00
  - early_treated=+1.00
  - a3=+1.00
  - s3_ind2=+0.90
  - max_ind2=+0.90
  - s3_ind1=+0.78

### NEGATIVE (feature near zero)
Example 1:
  - treatment_count=+1.00
  - early_treated=+1.00
  - switch_count=+1.00
  - a1=+1.00
  - max_ind1=+0.76
  - s1_ind1=+0.76
  - s2_ind1=+0.75
  - mean_ind1=+0.65
Example 2:
  - switch_count=+2.00
  - treatment_count=+1.00
  - a2=+1.00
  - max_ind1=+0.66
  - s1_ind1=+0.66
  - s2_ind1=+0.57
  - mean_ind1=+0.51
  - max_ind2=+0.42

--- feature 1 (density=0.426) ---
### POSITIVE (feature strongly active)
Example 1:
  - treatment_count=+2.00
  - switch_count=+2.00
  - a1=+1.00
  - a3=+1.00
  - early_treated=+1.00
  - s1_ind2=+0.94
  - max_ind2=+0.94
  - s3_ind1=+0.79
Example 2:
  - treatment_count=+2.00
  - switch_count=+1.00
  - a3=+1.00
  - a2=+1.00
  - max_ind2=+0.91
  - s3_ind2=+0.91
  - max_ind1=+0.86
  - s1_ind1=+0.86

### NEGATIVE (feature near zero)
Example 1:
  - treatment_count=+1.00
  - a1=+1.00
  - early_treated=+1.00
  - switch_count=+1.00
  - s2_ind1=+0.75
  - max_ind1=+0.75
  - delta1_ind1=+0.56
  - ind1_range=+0.56
Example 2:
  - treatment_count=+2.00
  - early_treated=+1.00
  - switch_count=+1.00
  - a2=+1.00
  - a1=+1.00
  - max_ind1=+0.77
  - s2_ind1=+0.77
  - s1_ind1=+0.60

--- feature 2 (density=0.475) ---
### POSITIVE (feature strongly active)
Example 1:
  - never_treated=+1.00
  - s3_ind2=+0.93
  - max_ind2=+0.93
  - mean_ind2=+0.76
  - s3_ind1=+0.75
  - max_ind1=+0.75
  - s1_ind2=+0.68
  - s2_ind2=+0.67
Example 2:
  - treatment_count=+1.00
  - switch_count=+1.00
  - a3=+1.00
  - max_ind1=+0.86
  - s3_ind1=+0.86
  - max_ind2=+0.80
  - s3_ind2=+0.80
  - s1_ind1=+0.76

### NEGATIVE (feature near zero)
Example 1:
  - treatment_count=+2.00
  - a1=+1.00
  - a2=+1.00
  - early_treated=+1.00
  - switch_count=+1.00
  - max_ind2=+0.77
  - s2_ind2=+0.77
  - s2_ind1=+0.71
Example 2:
  - switch_count=+2.00
  - treatment_count=+1.00
  - a2=+1.00
  - max_ind1=+0.88
  - s1_ind1=+0.88
  - max_ind2=+0.77
  - s1_ind2=+0.77
  - s2_ind2=+0.76
```

## 7. LLM client setup

```
LLM1/LLM2/Judge client: qwen-local
verifier backend: EmbeddingVerifier
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
all models warmed up
```

## 8a. LLM1 — pass 1 (cold start)

```
=== PASS 1: cold-start naming (no gold buffer) ===
  i0: "Treatment diversity and switches"  acc=0.50  rounds=1
  i1: "MIXED"  acc=0.50  rounds=1
  i2: "MIXED"  acc=0.50  rounds=1
  i3: "Ind2 Range > 0.5"  acc=0.88  rounds=0  -> pushed to gold buffer
  i4: "MIXED Delta Ind2 Range > 0.5 / Max Ind1 < 0.5"  acc=0.50  rounds=1
  i5: "MIXED"  acc=0.75  rounds=1  -> pushed to gold buffer
  i6: "Treatment diversity drop"  acc=0.31  rounds=1
  i8: "Max delta in indicators"  acc=0.62  rounds=1
  i13: "High switch and early treat"  acc=0.38  rounds=1
  i15: "MIXED"  acc=0.50  rounds=1
  i22: "MIXED"  acc=0.75  rounds=1  -> pushed to gold buffer
  i24: "MIXED"  acc=0.56  rounds=1
  i25: "Multiple late switches"  acc=0.38  rounds=1
  i26: "Low activation with few treatments"  acc=0.50  rounds=1
  i27: "MIXED high treatment and low switch count"  acc=0.50  rounds=1
  i29: "Treatments > Switches + Early Treated"  acc=0.44  rounds=1
  i30: "Treatment count increase"  acc=0.75  rounds=0  -> pushed to gold buffer

Pass 1 done in 299.9s
Gold buffer size: 4
Pass 2 will re-label 13 features (acc < 0.7).

=== PASS 2: re-naming low-acc features with gold-buffer few-shot ===
  i0: ""Max indicator value decrease""  acc=0.50  (was 0.50, Δ=+0.00, kept)
  i1: ""Max Ind2 > 0.75""  acc=0.50  (was 0.50, Δ=+0.00, kept)
  i2: ""Early treated + switch count""  acc=0.44  (was 0.50, Δ=-0.06, reverted)
  i4: "`Max Ind2 > 0.7`"  acc=0.75  (was 0.50, Δ=+0.25, kept)
  i6: ""Treatments per stage increase""  acc=0.75  (was 0.31, Δ=+0.44, kept)
  i8: ""Max Ind1 Range < 0.5""  acc=0.50  (was 0.62, Δ=-0.12, reverted)
  i13: ""Max Ind2 Range < 0.2""  acc=0.56  (was 0.38, Δ=+0.19, kept)
  i15: "`Switch Count > 1`"  acc=0.56  (was 0.50, Δ=+0.06, kept)
  i24: "Treatment count increase"  acc=0.88  (was 0.56, Δ=+0.31, kept)
  i25: ""Early treated increase""  acc=0.50  (was 0.38, Δ=+0.12, kept)
  i26: ""Max Ind1 < 0.9""  acc=0.62  (was 0.50, Δ=+0.12, kept)
  i27: ""Low switch_count range""  acc=0.50  (was 0.50, Δ=+0.00, kept)
  i29: "MIXED"  acc=0.81  (was 0.44, Δ=+0.38, kept)

Pass 2 done in 643.3s
Features pushed above alpha by gold buffer: 4 / 13

TOTAL LLM1 time: 943.2s
Final concept count above alpha: 8 / 17
```

## 8b. Feature polarity

```
computed polarity for 17 features
  HIGH-pushing: 7
  LOW-pushing:  8
  NEUTRAL:      2

Sample annotated dictionary entries:
  i0 -> ""Max indicator value decrease"" [→ LOW, strong, P(high|active)=0.08]
  i1 -> ""Max Ind2 > 0.75"" [→ LOW, strong, P(high|active)=0.07]
  i2 -> "MIXED" [→ LOW, strong, P(high|active)=0.23]
  i3 -> "Ind2 Range > 0.5" [→ HIGH, strong, P(high|active)=0.87]
  i4 -> "`Max Ind2 > 0.7`" [→ LOW, strong, P(high|active)=0.23]
```

## 8c. Snapshot save

```
saved snapshot to snapshot/run_001
PosixPath('snapshot/run_001')
```

## 9. Explanations on test inputs (LLM2 + Judge + K-selection)

```

=========================================================
input #0 (val idx 163)
  blackbox P(low, high) = (0.999, 0.001)
  active features = [0, 1, 2, 4, 13, 15, 26]
    i0: ""Max indicator value decrease"" [→ LOW, strong, P(high|active)=0.08]
    i1: ""Max Ind2 > 0.75"" [→ LOW, strong, P(high|active)=0.07]
    i2: "MIXED" [→ LOW, strong, P(high|active)=0.23]
    i4: "`Max Ind2 > 0.7`" [→ LOW, strong, P(high|active)=0.23]
    i13: ""Max Ind2 Range < 0.2"" [→ LOW, strong, P(high|active)=0.10]
    i15: "`Switch Count > 1`" [→ LOW, strong, P(high|active)=0.15]
    i26: ""Max Ind1 < 0.9"" [→ LOW, strong, P(high|active)=0.22]
  cand 0: KL=0.101  |S|=2  Len=2  total=0.401  Judge=(0.90,0.10) <-- WIN
  cand 1: KL=0.101  |S|=2  Len=2  total=0.401  Judge=(0.90,0.10) 
  cand 2: KL=0.101  |S|=2  Len=2  total=0.401  Judge=(0.90,0.10) 
  WINNING EXPLANATION:
  S: {i1, i15}
  Mechanisms:
  - [Max Ind2 > 0.7] [cite: i1] indicates that the maximum indicator value for Ind2 is greater than 0.7, which is associated with a LOW outcome.
  - [Switch Count > 1] [cite: i15] suggests that there have been more than one switch counts, which is also linked to a LOW outcome.
  Aggregation:
  - The cited cues support a LOW outcome. Both [Max Ind2 > 0.7] and [Switch Count > 1] contribute to the LOW signal, indicating that the model is likely predicting a LOW outcome.
  Limits:
  - Other indicators not considered here may influence the prediction. The provided concepts are incomplete and may not capture all relevant factors. [cite: i0, i2, i4, i13, i26]

=========================================================
input #1 (val idx 1869)
  blackbox P(low, high) = (0.474, 0.526)
  active features = [0, 4, 5, 13, 15, 22, 24]
    i0: ""Max indicator value decrease"" [→ LOW, strong, P(high|active)=0.08]
    i4: "`Max Ind2 > 0.7`" [→ LOW, strong, P(high|active)=0.23]
    i5: "MIXED" [→ HIGH, strong, P(high|active)=0.77]
    i13: ""Max Ind2 Range < 0.2"" [→ LOW, strong, P(high|active)=0.10]
    i15: "`Switch Count > 1`" [→ LOW, strong, P(high|active)=0.15]
    i22: "MIXED" [→ HIGH, strong, P(high|active)=0.75]
    i24: "Treatment count increase" [→ HIGH, strong, P(high|active)=0.83]
  cand 0: KL=8.044  |S|=2  Len=2  total=8.344  Judge=(0.00,1.00) 
  cand 1: KL=8.044  |S|=2  Len=2  total=8.344  Judge=(0.00,1.00) 
  cand 2: KL=0.756  |S|=2  Len=2  total=1.056  Judge=(0.05,0.95) <-- WIN
  WINNING EXPLANATION:
  S: {i5, i24}
  Mechanisms:
  - [i5] [cite: i5] is activated, indicating MIXED signals which supports a HIGH outcome.
  - [i24] [cite: i24] is activated, indicating a Treatment count increase, which supports a HIGH outcome.
  Aggregation:
  - Both cited features support a HIGH outcome. [cite: i5, i24]
  Limits:
  - Other features not considered might also influence the outcome. [cite: i0, i4, i13, i15, i22]
```

## 10. Final summary

```
#explained inputs: 2
mean winning KL(P_M || P_Judge): 0.428
mean cited features per winner: 2.0
mean mechanism count per winner: 2.0

verifier accuracy across 17 concepts: mean=0.66, min=0.50, max=0.88
concepts above alpha=0.7: 8 / 17
```

---

## Addendum: cherry-picking fix (polarity-aware K-candidate diversification)

A follow-up change in `pipeline.py` and `llm2_explainer.py` adds two things:

1. A **polarity summary hint** appended to the LLM2 user prompt that flags
   when the active features include both LOW- and HIGH-pushing concepts.
2. **Prescribed-S K-candidate strategies** — when polarities are mixed, the
   K candidates explicitly span `free` / `balanced` / `low_only` / `high_only`
   instead of just temperature-varied free choice.

Verified on the snapshot via `scripts/test_polarity_diversification.py`:

```
input #0 (val idx 163)
  blackbox P(low, high) = (0.999, 0.001)
  active LOW : [0, 1, 2, 4, 13, 15, 26]   active HIGH: []   (single polarity)
  All 4 candidates use 'free' strategy (auto-detected single polarity).
  KL = 0.101, Judge = (0.90, 0.10).   Same as before — no regression.

input #1 (val idx 1869)
  blackbox P(low, high) = (0.474, 0.526)
  active LOW : [0, 4, 13, 15]   active HIGH: [5, 22, 24]   (MIXED)
  [     free] cand 0: KL=0.067   Judge=(0.30,0.70)
  [ balanced] cand 1: KL=0.011   Judge=(0.40,0.60)   <-- WIN
  [ low_only] cand 2: KL=0.110   Judge=(0.70,0.30)
  [high_only] cand 3: KL=1.497   Judge=(0.01,0.99)
```

| metric | before fix | after fix |
| --- | ---:| ---:|
| input #1 winning KL | 0.756 | **0.011** |
| input #1 Judge probs | (0.05, 0.95) | (0.40, 0.60) |
| mean winning KL across both inputs | 0.428 | **0.056** |

The winning balanced explanation now openly says
"signals are mixed, leaning slightly LOW/HIGH" instead of cherry-picking
one polarity. The blackbox's near-50/50 uncertainty is faithfully transmitted
through to the Judge.

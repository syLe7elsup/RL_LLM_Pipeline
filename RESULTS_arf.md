# ARF (DQN-on-clinical-data) results

End-to-end run on the **toy ARF clinical simulator** via the
`Papillon-Xiang/LLM_RL` artifacts (stages 1-5: simulate → RL transitions
→ DQN → hidden states → SAE), with our `bb_pipeline` doing stages 6-8
(evidence + LLM1 + verifier + LLM2 + Judge + K-candidate selection).

Binary task: `y = (action_taken == IMV)` — "did the clinician escalate
to mechanical ventilation". Action 1 (HFNC) is <1% of training data, so
binary IMV-vs-other captures almost all signal.

Setup:
- 27 input features (clinical + demographic + derived), 3392 transitions
- DQN: 27 → 128 → 64 → 3 actions
- SAE: 64 → 128 (L1 sparsity), **38 / 128 alive features** with usable evidence
- LLM: Qwen 2.5-3B-Instruct (local MPS)
- Verifier: sentence-transformers/all-MiniLM-L6-v2 (per-concept threshold)
- 2-pass LLM1 with gold buffer few-shot

## LLM1: concept naming on real clinical features (very promising)

Total LLM1 time: **1505s** (38 features × 2 passes on Qwen 3B / MPS).

Final concepts ≥ α=0.65: **27 / 38 = 71%** (vs 47% on POLAR).

Examples — these are real clinical patterns:
- `i 67`: "SpO2_FiO2_ratio_HIGH"  (acc 1.00)
- `i104`: "MAP & PaO2 > 0"        (acc 1.00)
- `i 76`: "High SpO2_FiO2_ratio"  (acc 1.00)
- `i 4` : "PaO2 & PaCO2 negative" (acc 0.94)
- `i 17`: "shock_index_high_and_PaCO2_high" (acc 0.88)
- `i 66`: "PaO2 & CO2 Delta > 2.0" (acc 0.88)
- `i 42`: "PaO2 & HR negative"     (acc 0.81)

Pass 2 with gold buffer recovered 16 of 25 pass-1 failures into ≥α
(notably i20 0.50→0.94, i28 0.50→0.94, i58 0.50→0.94, i117 0.56→0.94).

## Stage 8: K-candidate explanations on N=10 val patients

| metric                     | value |
| -------------------------- | --:|
| argmax agreement rate      | **6 / 10 = 0.60** |
| KL  mean                   | 0.683 |
| KL  median                 | 0.288 |
| KL  max                    | 2.084 |

### Critically split by polarity composition

| subgroup            | n | argmax | KL median |
| ------------------- | -:| -----:| --------:|
| mixed polarity      | 7 | **0.857** | **0.146** |
| single-polarity all-LOW | 3 | **0.000** | **1.842** |

**Mixed-polarity inputs work as well on ARF as on POLAR.** The
polarity-aware K-candidate selection is genuinely doing its job:
across 7 mixed cases the `balanced` strategy wins 4 times and
`low_only`/`free` win the rest, and the explanation correctly
mirrors the blackbox's direction.

**The 3 failures are systematic:** all three patients have 23 active
SAE features, ALL with negative correlation to IMV (LOW polarity), yet
the DQN confidently predicts IMV (P > 0.9). These appear to be
critically-ill patients where the **combination** of normally-LOW
features signals deterioration — a non-monotonic interaction that our
marginal polarity computation cannot capture.

```
vidx   blackbox        judge          KL     L/H    strategy
 50    (0.10, 0.90)   (0.90, 0.10)   1.756   23/0   free       ❌
 52    (0.08, 0.92)   (0.90, 0.10)   1.842   23/0   free       ❌
 55    (0.03, 0.97)   (0.90, 0.10)   2.084   23/0   free       ❌
```

When all candidates yield identical Judge probabilities (because they
all use the same single-polarity feature set), the K-candidate
mechanism has no recourse and the polarity hint dominates LLM2 toward
the wrong direction.

## Comparison with POLAR

| dimension              | POLAR (N=50)       | ARF (N=10)         |
| ---------------------- | ------------------:| ------------------:|
| argmax agreement       | 0.980              | 0.600              |
| KL median              | 0.024              | 0.288              |
| KL max                 | 0.165              | 2.084              |
| concepts ≥ α           | 8/17 = 47%         | 27/38 = 71%        |
| single-polarity failure| ~none              | 3/3 = 100%         |

LLM1 quality is much higher on ARF (clinical features → real medical
concepts). But the single-polarity failure mode — which barely existed
on POLAR's lower-dim 3-stage data — dominates the ARF failure budget.

## Next steps to address single-polarity failures

1. **Per-input feature attribution** — replace marginal polarity with
   gradient × activation or SHAP on the DQN's IMV logit. This gives
   per-input direction signal that captures non-monotonic combinations.
2. **Show LLM2 the blackbox prediction** as a side-channel hint, with
   a note "polarity hints suggest X but model says Y" when they conflict.
   (Risk: contaminates the Judge step.)
3. **Polarity confidence weighting** — when polarity is uniform across
   active features, lower the confidence of the polarity hint in LLM2's
   prompt rather than asserting it.
4. **3-class generalization** — collapsing 3 actions to binary may be
   hiding the structure DQN uses; the all-LOW failure mode might
   correspond to the IMV-vs-HFNC boundary which doesn't exist binary-wise.

Full per-input results: `arf_snapshot/run_001/explanations_n10.json`.
LLM1 concepts + verifier scores: `arf_snapshot/run_001/concepts.json`.

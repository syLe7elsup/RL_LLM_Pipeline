"""Toy synthetic Acute Respiratory Failure (ARF) ICU data generator.

This is a synthetic but clinically plausible longitudinal dataset, intended
as a more realistic test bed for the SAE-based blackbox-explanation pipeline
than POLAR's abstract 2D state space. It supplements rather than replaces
``polar_data.py``.

Source: provided by the user (toy ARF simulator). Embedded here verbatim
with light wrapping so that the rest of bb_pipeline can call
``generate_arf_data(...)`` and get back the three dataframes (long, concept,
patient) directly without going through CSVs.

Each patient has 16-48 hourly time steps. At each step we observe ~17
clinical variables driven by 6 latent concepts (oxygenation_failure,
ventilatory_failure, metabolic_stress, hemodynamic_instability,
inflammation_severity, recovery_trend). The target is binary
``deterioration_next``: whether the patient will deteriorate in the
following step.

Why this matters: the latent concepts are emitted as ground truth
(``concept_df``), so we can quantitatively check whether our SAE's sparse
features actually align with the real underlying latent structure — a
validation we cannot run on POLAR's data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility functions


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _clip(x, low, high):
    return np.minimum(np.maximum(x, low), high)


# ---------------------------------------------------------------------------
# Main simulator (verbatim from user's `simulate_toy_arf_data`, light style edits)


def simulate_toy_arf_data(
    n_patients: int = 1000,
    min_time: int = 12,
    max_time: int = 72,
    seed: int = 42,
    include_missingness: bool = True,
    missing_rate_base: float = 0.03,
):
    """Simulate longitudinal toy ARF data.

    Returns three pandas DataFrames:
        long_df    -- one row per (patient_id, time_step), observed variables
        concept_df -- one row per (patient_id, time_step), 6 ground-truth
                      latent concepts
        patient_df -- one row per patient, static demographics + traj_len
    """

    rng = np.random.default_rng(seed)

    long_rows: list[dict] = []
    concept_rows: list[dict] = []
    patient_rows: list[dict] = []

    for pid in range(n_patients):
        T = int(rng.integers(min_time, max_time + 1))

        age = _clip(rng.normal(65, 15), 18, 95)
        bmi = _clip(rng.normal(29, 7), 16, 55)
        copd = rng.binomial(1, 0.22)
        chf = rng.binomial(1, 0.18)
        ckd = rng.binomial(1, 0.15)
        diabetes = rng.binomial(1, 0.28)
        obesity = 1 if bmi >= 30 else 0
        # 0 = mainly hypoxemic, 1 = mixed, 2 = hypercapnic-predominant
        subtype = rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20])

        # Initial latent state.
        c1 = _clip(rng.normal(0.7 if subtype in [0, 1] else 0.3, 0.45), -1.5, 2.5)
        c2 = _clip(rng.normal(0.9 if subtype == 2 else 0.25, 0.45), -1.5, 2.5)
        c3 = _clip(rng.normal(0.35, 0.5), -1.5, 2.5)
        c4 = _clip(rng.normal(0.2 + 0.25 * ckd + 0.2 * chf, 0.45), -1.5, 2.5)
        c5 = _clip(rng.normal(0.6, 0.55), -1.5, 2.5)
        c6 = _clip(rng.normal(-0.2, 0.5), -2.5, 2.5)

        patient_rows.append(
            dict(
                patient_id=pid, age=age, bmi=bmi, copd=copd, chf=chf, ckd=ckd,
                diabetes=diabetes, obesity=obesity, subtype=subtype, traj_len=T,
            )
        )

        prev_action = 0

        for t in range(T):
            # Latent dynamics
            c6 = _clip(0.82 * c6 + rng.normal(0, 0.18) - 0.05 * (c1 + c2 + c4)
                       + 0.03 * (t / max(T, 1)), -2.5, 2.5)
            c5 = _clip(0.92 * c5 + rng.normal(0, 0.12) - 0.08 * c6, -1.5, 3.0)
            c1 = _clip(0.80 * c1 + 0.16 * c5 - 0.22 * c6 + rng.normal(0, 0.15), -1.5, 3.0)
            c2 = _clip(0.82 * c2 + 0.18 * copd + 0.08 * (subtype == 2) - 0.12 * c6
                       + 0.05 * c1 + rng.normal(0, 0.14), -1.5, 3.0)
            c3 = _clip(0.76 * c3 + 0.18 * c4 + 0.10 * c5 - 0.14 * c6
                       + rng.normal(0, 0.16), -1.5, 3.0)
            c4 = _clip(0.78 * c4 + 0.12 * c5 + 0.10 * c3 + 0.07 * chf + 0.05 * ckd
                       - 0.10 * c6 + rng.normal(0, 0.16), -1.5, 3.0)

            # Observed variables
            fio2 = _clip(0.24 + 0.12 * c1 + 0.04 * np.maximum(c2, 0) - 0.03 * c6
                         + rng.normal(0, 0.03), 0.21, 1.0)
            pao2 = _clip(95 - 22 * c1 - 5 * np.maximum(c2, 0) + 4 * c6 + rng.normal(0, 6), 35, 160)
            spo2 = _clip(96 - 4.8 * c1 - 0.8 * np.maximum(c2, 0) + 0.8 * c6 + rng.normal(0, 1.4), 70, 100)
            rr = _clip(18 + 5.0 * c1 + 4.0 * c2 + 2.2 * c3 - 1.5 * c6 + rng.normal(0, 2.0), 8, 50)
            paco2 = _clip(40 + 8.5 * c2 + 1.8 * c3 - 1.0 * c6 + 2.5 * copd + rng.normal(0, 3.0), 20, 100)
            ph = _clip(7.40 - 0.045 * c2 - 0.055 * c3 + 0.010 * c6 + rng.normal(0, 0.02), 6.95, 7.55)
            map_val = _clip(83 - 9.0 * c4 - 2.0 * c3 + 1.5 * c6 + rng.normal(0, 4.0), 35, 130)
            hr = _clip(88 + 7.0 * c4 + 3.0 * c5 + 2.0 * c3 + 1.5 * c1 - 1.0 * c6 + rng.normal(0, 5.0), 40, 180)
            lactate = _clip(1.3 + 0.9 * np.maximum(c4, 0) + 0.7 * np.maximum(c3, 0)
                            + 0.15 * c5 + rng.normal(0, 0.35), 0.4, 12.0)
            urine_output = _clip(55 - 10 * np.maximum(c4, 0) - 4 * np.maximum(c3, 0)
                                 + 3 * c6 + rng.normal(0, 7), 0, 120)
            temp = _clip(37.0 + 0.55 * c5 + 0.08 * c4 + rng.normal(0, 0.25), 35.0, 41.0)
            crp = _clip(12 + 24 * np.maximum(c5, 0) + 4 * np.maximum(c1, 0) + rng.normal(0, 8), 0, 300)
            wbc = _clip(7.5 + 2.2 * c5 + 0.5 * c3 + rng.normal(0, 1.6), 1.0, 35.0)
            mental_status_score = _clip(
                15 - 0.9 * np.maximum(c2, 0) - 0.8 * np.maximum(c4, 0)
                - 0.5 * np.maximum(c3, 0) + 0.3 * c6 + rng.normal(0, 0.6),
                3, 15,
            )

            vasopressor_prob = _sigmoid(-2.2 + 1.4 * c4 + 0.4 * c3 + 0.2 * c5)
            vasopressor_use = rng.binomial(1, vasopressor_prob)

            # Clinician-like support decision
            imv_score = (-4.2 + 1.45 * c1 + 1.35 * c2 + 1.00 * c3 + 0.55 * c4
                         + 0.30 * c5 - 0.65 * c6
                         + 0.9 * (ph < 7.25) + 0.8 * (mental_status_score < 12)
                         + 0.5 * (spo2 < 88) + 0.3 * (rr > 32))
            hfnc_niv_score = (-1.8 + 1.25 * c1 + 0.65 * c2 + 0.20 * c3 - 0.20 * c4
                              + 0.15 * c5 - 0.45 * c6
                              + 0.4 * (spo2 < 92) + 0.25 * (fio2 > 0.4))

            p_imv = _sigmoid(imv_score)
            p_hfnc = _sigmoid(hfnc_niv_score) * (1 - p_imv)
            if prev_action == 2:
                p_imv = min(0.97, p_imv + 0.20)
            elif prev_action == 1:
                p_hfnc = min(0.95, p_hfnc + 0.12)
            p_std = max(0.0, 1 - p_imv - p_hfnc)
            probs = np.array([p_std, p_hfnc, p_imv], dtype=float)
            probs = probs / probs.sum()
            action = int(rng.choice([0, 1, 2], p=probs))
            support_device = ["standard_oxygen", "hfnc_or_niv", "imv"][action]

            deterioration_logit = (-3.2 + 1.25 * c1 + 1.10 * c2 + 0.95 * c3 + 1.00 * c4
                                   + 0.55 * c5 - 0.90 * c6
                                   + 0.45 * (lactate > 2.5) + 0.45 * (map_val < 65)
                                   + 0.55 * (ph < 7.30) + 0.45 * (spo2 < 90))
            deterioration_prob = _sigmoid(deterioration_logit)
            deterioration_next = int(rng.binomial(1, deterioration_prob))

            row = dict(
                patient_id=pid, time_step=t,
                age=age, bmi=bmi, copd=copd, chf=chf, ckd=ckd, diabetes=diabetes,
                obesity=obesity, subtype=subtype,
                SpO2=spo2, FiO2=fio2, PaO2=pao2, RR=rr, PaCO2=paco2, pH=ph,
                MAP=map_val, HR=hr, lactate=lactate, temp=temp, CRP=crp, WBC=wbc,
                mental_status_score=mental_status_score, urine_output=urine_output,
                vasopressor_use=vasopressor_use, support_device=support_device,
                action=action, deterioration_next=deterioration_next,
                deterioration_prob=deterioration_prob,
            )
            concept_rows.append(dict(
                patient_id=pid, time_step=t,
                oxygenation_failure=c1, ventilatory_failure=c2,
                metabolic_stress=c3, hemodynamic_instability=c4,
                inflammation_severity=c5, recovery_trend=c6,
            ))

            if include_missingness:
                abg_missing_prob = _clip(missing_rate_base + 0.15 * (action == 0)
                                          + 0.05 * (t > 0 and t % 6 != 0), 0, 0.5)
                lab_missing_prob = _clip(missing_rate_base + 0.06 * (t > 0 and t % 4 != 0), 0, 0.35)
                if rng.random() < abg_missing_prob: row["PaO2"] = np.nan
                if rng.random() < abg_missing_prob: row["PaCO2"] = np.nan
                if rng.random() < abg_missing_prob: row["pH"] = np.nan
                if rng.random() < lab_missing_prob: row["CRP"] = np.nan
                if rng.random() < lab_missing_prob: row["WBC"] = np.nan
                if rng.random() < lab_missing_prob: row["lactate"] = np.nan

            long_rows.append(row)
            prev_action = action

    long_df = pd.DataFrame(long_rows)
    concept_df = pd.DataFrame(concept_rows)
    patient_df = pd.DataFrame(patient_rows)

    # Derived clinically meaningful variables
    long_df["SpO2_FiO2_ratio"] = long_df["SpO2"] / long_df["FiO2"]
    long_df["shock_index"] = long_df["HR"] / long_df["MAP"]
    long_df["acidotic"] = (long_df["pH"] < 7.30).astype(float)
    long_df["severe_hypoxemia_like"] = (
        (long_df["SpO2"] < 90) & (long_df["FiO2"] > 0.6)
    ).astype(float)

    return long_df, concept_df, patient_df


# Names of the 6 latent ground-truth concepts (for concept_grounding.py).
GROUND_TRUTH_CONCEPTS = [
    "oxygenation_failure",
    "ventilatory_failure",
    "metabolic_stress",
    "hemodynamic_instability",
    "inflammation_severity",
    "recovery_trend",
]

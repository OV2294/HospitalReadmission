"""
Generates a small synthetic sample that matches the exact column schema of
the real UCI "Diabetes 130-US Hospitals" dataset, purely so unit tests run
fast without needing the full 101,766-row / 19MB CSV checked out.

This is test-only scaffolding — the actual project pipeline (src/) always
runs against data/raw/readmission_raw.csv, the real dataset.
"""

import numpy as np
import pandas as pd

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def make_sample_raw_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ages = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]

    df = pd.DataFrame({
        "encounter_id": np.arange(1, n + 1),
        "patient_nbr": rng.integers(1, n // 2, n),  # some repeated patients
        "race": rng.choice(["Caucasian", "AfricanAmerican", "?", "Asian", "Hispanic"], n),
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.choice(ages, n),
        "weight": "?",
        "admission_type_id": rng.integers(1, 9, n),
        "discharge_disposition_id": rng.choice(
            [1, 2, 3, 6, 11, 18], n, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]
        ),
        "admission_source_id": rng.integers(1, 18, n),
        "time_in_hospital": rng.integers(1, 15, n),
        "payer_code": "?",
        "medical_specialty": "?",
        "num_lab_procedures": rng.integers(1, 100, n),
        "num_procedures": rng.integers(0, 6, n),
        "num_medications": rng.integers(1, 40, n),
        "number_outpatient": rng.poisson(0.3, n),
        "number_emergency": rng.poisson(0.2, n),
        "number_inpatient": rng.poisson(0.4, n),
        "diag_1": rng.choice(["250.83", "414", "428", "486", "V45", "780"], n),
        "diag_2": rng.choice(["250", "401", "496"], n),
        "diag_3": rng.choice(["250", "272", "V10"], n),
        "number_diagnoses": rng.integers(1, 16, n),
        "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"], n, p=[0.8, 0.1, 0.05, 0.05]),
        "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n, p=[0.75, 0.1, 0.08, 0.07]),
        "change": rng.choice(["No", "Ch"], n),
        "diabetesMed": rng.choice(["No", "Yes"], n),
        "readmitted": rng.choice(["NO", ">30", "<30"], n, p=[0.6, 0.25, 0.15]),
    })

    for col in MED_COLS:
        df[col] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.7, 0.2, 0.05, 0.05])

    return df

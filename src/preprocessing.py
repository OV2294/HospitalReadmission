"""
Preprocessing / feature engineering stage for the REAL UCI "Diabetes
130-US Hospitals for Years 1999-2008" dataset (101,766 encounters, 50 raw
columns; UCI ML Repository dataset #296).

Cleaning decisions (standard practice for this dataset, matching the
original Strack et al. 2014 study and most published work on it):

  - '?' placeholder values are treated as missing.
  - Rows discharged to expired/hospice (a patient who died or entered
    hospice cannot be "readmitted") are excluded.
  - Only the FIRST encounter per patient is kept, to avoid data leakage
    from a single patient appearing multiple times.
  - Columns with extreme missingness (weight, payer_code,
    medical_specialty) are dropped.
  - Zero-variance columns (e.g. examide, citoglipton — every patient has
    the same value) are dropped automatically.
  - The primary diagnosis code (diag_1, an ICD-9 code) is bucketed into
    9 clinically meaningful categories rather than used as 700+ raw codes.
  - diag_2 / diag_3 (secondary/tertiary diagnoses) are dropped for this
    mini-project's scope; they are natural future-work additions.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ID_COLS_TO_DROP = ["encounter_id", "patient_nbr"]
HIGH_MISSING_COLS = ["weight", "payer_code", "medical_specialty"]
RAW_DIAGNOSIS_COLS = ["diag_2", "diag_3"]  # diag_1 is kept, bucketed
TARGET_COL = "readmitted"
EXPIRED_HOSPICE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]


def bucket_icd9(code):
    """Map an ICD-9 diagnosis code to one of 9 clinical categories used
    throughout the published literature on this dataset."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"

    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if 250 <= val < 251:
        return "Diabetes"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


def clean(df: pd.DataFrame):
    df = df.copy()
    df = df.replace("?", np.nan)

    # Drop expired/hospice discharges — not valid readmission candidates
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_DISPOSITION_IDS)]

    # Keep only the first encounter per patient to avoid leakage
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    # Drop a handful of malformed gender rows ("Unknown/Invalid")
    df = df[df["gender"].isin(["Male", "Female"])]

    df["diag_1_category"] = df["diag_1"].apply(bucket_icd9)

    drop_cols = ID_COLS_TO_DROP + HIGH_MISSING_COLS + RAW_DIAGNOSIS_COLS + ["diag_1"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop any remaining zero-variance columns automatically
    nunique = df.nunique()
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)

    return df, zero_var_cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    return df


def preprocess(input_path: str, output_path: str, encoder_path: str):
    raw = pd.read_csv(input_path)
    df, dropped_zero_var = clean(raw)
    df = engineer_features(df)

    df["label"] = (df[TARGET_COL] == "<30").astype(int)
    df = df.drop(columns=[TARGET_COL])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "label"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["label"]]
    feature_order = [c for c in df.columns if c != "label"]

    # Any remaining missing values in categoricals become their own category
    df[categorical_cols] = df[categorical_cols].fillna("Missing")

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(
        {
            "encoder": encoder,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "feature_order": feature_order,
            "dropped_zero_variance_cols": dropped_zero_var,
        },
        encoder_path,
    )

    print(f"Raw rows: {len(raw)} -> after cleaning/dedup: {len(df)}")
    print(f"Dropped zero-variance columns: {dropped_zero_var}")
    print(f"Final feature count: {len(categorical_cols) + len(numeric_cols)} "
          f"({len(categorical_cols)} categorical, {len(numeric_cols)} numeric)")
    print(f"Label balance:\n{df['label'].value_counts(normalize=True)}")
    print(f"Wrote {len(df)} processed rows to {output_path}")
    print(f"Saved fitted encoder to {encoder_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/readmission_raw.csv")
    parser.add_argument("--output", default="data/processed/readmission_features.csv")
    parser.add_argument("--encoder-out", default="models/preprocessor.pkl")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.encoder_out)

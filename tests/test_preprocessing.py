import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import bucket_icd9, engineer_features, preprocess  # noqa: E402
from tests.fixtures import make_sample_raw_df  # noqa: E402


def test_bucket_icd9_known_ranges():
    assert bucket_icd9("410") == "Circulatory"
    assert bucket_icd9("486") == "Respiratory"
    assert bucket_icd9("250.83") == "Diabetes"
    assert bucket_icd9("820") == "Injury"
    assert bucket_icd9("V45") == "Other"
    assert bucket_icd9(None) == "Missing"


def test_engineer_features_adds_total_prior_visits():
    df = pd.DataFrame({
        "number_outpatient": [1, 0],
        "number_emergency": [2, 0],
        "number_inpatient": [0, 3],
    })
    out = engineer_features(df)
    assert "total_prior_visits" in out.columns
    assert list(out["total_prior_visits"]) == [3, 3]


def test_preprocess_end_to_end(tmp_path):
    raw = make_sample_raw_df(300)
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "features.csv"
    encoder_path = tmp_path / "encoder.pkl"
    raw.to_csv(raw_path, index=False)

    df = preprocess(str(raw_path), str(out_path), str(encoder_path))

    assert "label" in df.columns
    assert "readmitted" not in df.columns
    assert "patient_nbr" not in df.columns  # ID column dropped
    assert "weight" not in df.columns       # high-missingness column dropped
    assert "diag_1" not in df.columns       # replaced by diag_1_category
    assert set(df["label"].unique()) <= {0, 1}
    assert os.path.exists(out_path)
    assert os.path.exists(encoder_path)

    # every remaining column must be numeric after encoding
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} was not encoded"


def test_preprocess_drops_expired_hospice_rows(tmp_path):
    raw = make_sample_raw_df(300)
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "features.csv"
    encoder_path = tmp_path / "encoder.pkl"
    raw.to_csv(raw_path, index=False)

    df = preprocess(str(raw_path), str(out_path), str(encoder_path))
    # After dedup + expired/hospice removal, we should have <= unique patients
    assert len(df) <= raw["patient_nbr"].nunique()

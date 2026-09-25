import os
import sys

import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess  # noqa: E402
from src.train import main as train_main  # noqa: E402
from tests.fixtures import make_sample_raw_df  # noqa: E402


def test_training_smoke(tmp_path, monkeypatch):
    """Runs the full mini pipeline on a small synthetic-but-real-schema sample
    and checks that every model produces sane, bounded metrics."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")

    raw = make_sample_raw_df(400)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    raw.to_csv("data/raw/readmission_raw.csv", index=False)
    preprocess(
        "data/raw/readmission_raw.csv",
        "data/processed/readmission_features.csv",
        "models/preprocessor.pkl",
    )

    results = train_main("data/processed/readmission_features.csv")

    assert set(results.keys()) == {
        "logistic_regression", "random_forest", "xgboost", "lightgbm",
    }
    for name, metrics in results.items():
        for metric_name in ("accuracy", "precision", "recall", "f1", "auc"):
            assert 0.0 <= metrics[metric_name] <= 1.0, f"{name}.{metric_name} out of bounds"

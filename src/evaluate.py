"""
Performance gate used by the CI and retraining pipelines: fails (non-zero
exit code) if the best available MLflow run does not meet a minimum recall,
preventing a bad model from being registered or deployed.
"""

import argparse
import sys

import mlflow

EXPERIMENT_NAME = "hospital-readmission-risk"


def check_best_run(min_recall: float) -> bool:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.recall DESC"],
    )
    if runs.empty:
        print("No runs found in MLflow experiment.")
        return False

    best = runs.iloc[0]
    recall = best["metrics.recall"]
    auc = best["metrics.auc"]
    print(f"Best run: {best['tags.mlflow.runName']}  recall={recall:.3f}  auc={auc:.3f}")

    if recall < min_recall:
        print(f"FAIL: recall {recall:.3f} is below required minimum {min_recall}")
        return False

    print(f"PASS: recall {recall:.3f} meets required minimum {min_recall}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-recall", type=float, default=0.45)
    args = parser.parse_args()
    ok = check_best_run(args.min_recall)
    sys.exit(0 if ok else 1)

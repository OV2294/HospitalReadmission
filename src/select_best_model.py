"""
Queries all MLflow runs in the experiment, ranks them by recall (primary,
since a false negative is clinically costlier) and AUC-ROC (secondary),
registers the winner in the MLflow Model Registry, promotes it to the
"Production" alias/stage, and exports it as models/best_model.pkl for the
Docker prediction service to load.

Also enforces a minimum-performance gate: a candidate is only promoted if it
beats (or ties) the currently registered Production model, so an automatic
retraining run can never silently regress the deployed model.
"""

import argparse
import shutil

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "hospital-readmission-risk"
REGISTERED_MODEL_NAME = "readmission-risk-model"


def get_current_production_metrics(client: MlflowClient):
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    except Exception:
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics


def select_and_register(min_recall: float = 0.45):
    client = MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.recall DESC", "metrics.auc DESC"],
    )
    if runs.empty:
        raise SystemExit("No MLflow runs found. Run src/train.py first.")

    best = runs.iloc[0]
    print(f"Best run: {best['tags.mlflow.runName']}  "
          f"recall={best['metrics.recall']:.3f}  auc={best['metrics.auc']:.3f}")

    if best["metrics.recall"] < min_recall:
        raise SystemExit(
            f"Best candidate recall {best['metrics.recall']:.3f} is below the "
            f"minimum threshold {min_recall}. Refusing to register/promote."
        )

    current_prod = get_current_production_metrics(client)
    if current_prod is not None:
        print(f"Current production: recall={current_prod.get('recall', 0):.3f}  "
              f"auc={current_prod.get('auc', 0):.3f}")
        if best["metrics.recall"] < current_prod.get("recall", 0):
            print("New candidate does NOT beat current production model. "
                  "Keeping existing production model (no promotion).")
            return None

    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Registered version {mv.version} of '{REGISTERED_MODEL_NAME}' and "
          f"promoted to Production.")

    # Export a flat copy for the Docker service to load without needing
    # an MLflow tracking server at inference time.
    local_path = mlflow.artifacts.download_artifacts(f"{model_uri}/model.pkl")
    shutil.copy(local_path, "models/best_model.pkl")
    print("Exported production model to models/best_model.pkl")
    return mv.version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-recall", type=float, default=0.45)
    args = parser.parse_args()
    select_and_register(min_recall=args.min_recall)

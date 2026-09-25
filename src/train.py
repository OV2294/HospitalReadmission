"""
Trains multiple candidate models on the processed dataset and logs every run
(params, metrics, model artifact) to MLflow under a single experiment so they
can be compared side by side and the best one selected later.

The real dataset is heavily imbalanced (~9% positive class for 30-day
readmission), so every model is trained with class-balancing enabled
(class_weight="balanced" for sklearn/LightGBM, scale_pos_weight for
XGBoost) — otherwise a model could get >90% accuracy by simply predicting
"not readmitted" for everyone, which is useless in practice and would
score terribly on recall, the metric we actually care about here.
"""

import argparse

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "hospital-readmission-risk"
FEATURES_PATH_DEFAULT = "data/processed/readmission_features.csv"


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def log_and_eval(run_name, model, X_train, y_train, X_test, y_test, params):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
        }

        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(
            model, "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        print(f"[{run_name}] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        return metrics


def main(features_path):
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = load_data(features_path)

    # scale_pos_weight for boosted-tree libraries that don't take
    # class_weight="balanced" directly
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    results = {}

    params_lr = {"model": "logistic_regression", "max_iter": 1000, "class_weight": "balanced"}
    results["logistic_regression"] = log_and_eval(
        "logistic_regression",
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X_train, y_train, X_test, y_test, params_lr,
    )

    params_rf = {"model": "random_forest", "n_estimators": 300, "max_depth": 12,
                 "class_weight": "balanced"}
    results["random_forest"] = log_and_eval(
        "random_forest",
        RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=42,
        ),
        X_train, y_train, X_test, y_test, params_rf,
    )

    params_xgb = {"model": "xgboost", "n_estimators": 300, "max_depth": 6,
                  "learning_rate": 0.1, "scale_pos_weight": round(scale_pos_weight, 3)}
    results["xgboost"] = log_and_eval(
        "xgboost",
        xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42,
        ),
        X_train, y_train, X_test, y_test, params_xgb,
    )

    params_lgbm = {"model": "lightgbm", "n_estimators": 300, "max_depth": -1,
                   "learning_rate": 0.1, "class_weight": "balanced"}
    results["lightgbm"] = log_and_eval(
        "lightgbm",
        lgb.LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
        X_train, y_train, X_test, y_test, params_lgbm,
    )

    print("\nSummary:")
    for name, m in results.items():
        print(f"  {name:20s} recall={m['recall']:.3f}  auc={m['auc']:.3f}  f1={m['f1']:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=FEATURES_PATH_DEFAULT)
    args = parser.parse_args()
    main(args.features)

import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import PatientData, PredictionResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "models/preprocessor.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

app = FastAPI(
    title="Hospital Readmission Risk Prediction API",
    description="Predicts the probability that a patient will be readmitted within 30 days.",
    version="1.0.0",
)

_model = None
_preprocessor_bundle = None


def load_artifacts():
    global _model, _preprocessor_bundle
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    if _preprocessor_bundle is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise RuntimeError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
        _preprocessor_bundle = joblib.load(PREPROCESSOR_PATH)
    return _model, _preprocessor_bundle


def build_feature_row(patient: PatientData) -> pd.DataFrame:
    _, bundle = load_artifacts()
    encoder = bundle["encoder"]
    categorical_cols = bundle["categorical_cols"]

    row = patient.model_dump(by_alias=True)

    row["number_outpatient"] = patient.number_outpatient
    row["number_emergency"] = patient.number_emergency
    row["number_inpatient"] = patient.number_inpatient
    row["total_prior_visits"] = (
        patient.number_outpatient + patient.number_emergency + patient.number_inpatient
    )

    df = pd.DataFrame([row])
    # Keep only the columns the fitted encoder/model actually expects, in the
    # exact order used during training
    feature_order = bundle["feature_order"]
    df[categorical_cols] = df[categorical_cols].astype(str)
    df[categorical_cols] = encoder.transform(df[categorical_cols])
    df = df[feature_order]
    return df


@app.on_event("startup")
def _startup():
    try:
        load_artifacts()
    except RuntimeError as e:
        # Don't crash the container on startup if artifacts aren't mounted yet;
        # /health will report the problem and /predict will raise a clear 503.
        print(f"Warning: {e}")


@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)
    return {"status": "ok" if ok else "model_not_loaded", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    try:
        model, _ = load_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = build_feature_row(patient)
    risk_prob = float(model.predict_proba(features)[0][1])

    return PredictionResponse(
        readmission_risk=round(risk_prob, 3),
        risk_level="High" if risk_prob > 0.5 else "Low",
        model_version=MODEL_VERSION,
    )

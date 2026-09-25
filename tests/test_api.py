import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app  # noqa: E402

client = TestClient(app)

SAMPLE_PATIENT = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 8,
    "num_lab_procedures": 60,
    "num_procedures": 2,
    "num_medications": 25,
    "number_outpatient": 1,
    "number_emergency": 2,
    "number_inpatient": 3,
    "number_diagnoses": 9,
    "diag_1_category": "Circulatory",
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "insulin": "Up",
    "change": "Ch",
    "diabetesMed": "Yes",
}


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_predict_endpoint_returns_valid_response():
    resp = client.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["readmission_risk"] <= 1.0
    assert body["risk_level"] in {"High", "Low"}


def test_predict_endpoint_rejects_missing_fields():
    incomplete = {"race": "Caucasian"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_uses_medication_defaults():
    """Medication fields default to 'No' so callers don't have to specify
    all 20 of them for a quick test."""
    minimal = dict(SAMPLE_PATIENT)
    resp = client.post("/predict", json=minimal)
    assert resp.status_code == 200

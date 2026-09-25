"""
Hospital Readmission Risk Prediction — Streamlit Demo
-------------------------------------------------------
Loads the trained model + preprocessor produced by this project's
DVC/MLflow pipeline (models/best_model.pkl, models/preprocessor.pkl)
and serves the exact same prediction logic as api/main.py, in an
interactive UI instead of a REST endpoint.
"""

import os

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "models", "preprocessor.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    bundle = joblib.load(PREPROCESSOR_PATH)
    return model, bundle


model, bundle = load_artifacts()
encoder = bundle["encoder"]
categorical_cols = bundle["categorical_cols"]
feature_order = bundle["feature_order"]

MED_OPTIONS = ["No", "Steady", "Up", "Down"]
MED_FIELDS = [
    ("metformin", "Metformin"),
    ("repaglinide", "Repaglinide"),
    ("nateglinide", "Nateglinide"),
    ("chlorpropamide", "Chlorpropamide"),
    ("glimepiride", "Glimepiride"),
    ("acetohexamide", "Acetohexamide"),
    ("glipizide", "Glipizide"),
    ("glyburide", "Glyburide"),
    ("tolbutamide", "Tolbutamide"),
    ("pioglitazone", "Pioglitazone"),
    ("rosiglitazone", "Rosiglitazone"),
    ("acarbose", "Acarbose"),
    ("miglitol", "Miglitol"),
    ("troglitazone", "Troglitazone"),
    ("tolazamide", "Tolazamide"),
    ("insulin", "Insulin"),
    ("glyburide-metformin", "Glyburide-Metformin"),
    ("glipizide-metformin", "Glipizide-Metformin"),
    ("metformin-rosiglitazone", "Metformin-Rosiglitazone"),
    ("metformin-pioglitazone", "Metformin-Pioglitazone"),
]

DIAG_CATEGORIES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]

# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
st.title("🏥 Hospital Readmission Risk Predictor")
st.caption(
    "Predicts the probability that a patient will be readmitted within "
    "30 days of discharge — trained on the UCI Diabetes 130-US Hospitals "
    "dataset (101,766 encounters, 1999–2008)."
)

with st.expander("ℹ️ About this model", expanded=False):
    st.markdown(
        """
- **Model:** Logistic Regression (`class_weight="balanced"`), selected over
  Random Forest, XGBoost and LightGBM because it achieves the **highest recall**
  — clinically the most important metric here, since missing a high-risk
  patient (a false negative) is costlier than a false alarm.
- **Reported performance:** Accuracy 0.638 · Precision 0.129 · **Recall 0.529**
  · F1 0.208 · AUC-ROC 0.626
- An AUC around 0.62–0.63 matches published results on this exact dataset
  (Strack et al., 2014). Roughly 9% of real encounters are positive
  (readmitted within 30 days), so these numbers reflect a genuinely hard,
  imbalanced clinical problem — not a bug.
"""
    )

st.divider()

# --------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------
st.subheader("Patient encounter details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    race = st.selectbox(
        "Race",
        ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"],
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.selectbox(
        "Age bracket",
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
         "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        index=7,
    )

with col2:
    st.markdown("**Admission / discharge (UCI numeric IDs)**")
    admission_type_id = st.number_input("Admission type ID", min_value=1, max_value=8, value=1)
    discharge_disposition_id = st.number_input(
        "Discharge disposition ID", min_value=1, max_value=28, value=1
    )
    admission_source_id = st.number_input(
        "Admission source ID", min_value=1, max_value=25, value=7
    )
    diag_1_category = st.selectbox("Primary diagnosis category", DIAG_CATEGORIES)

with col3:
    st.markdown("**Encounter counts**")
    time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 5)
    num_lab_procedures = st.number_input("Num. lab procedures", min_value=0, value=45)
    num_procedures = st.number_input("Num. procedures", min_value=0, value=1)
    num_medications = st.number_input("Num. medications", min_value=0, value=18)
    number_diagnoses = st.number_input("Num. diagnoses", min_value=1, value=9)

st.markdown("**Prior visit history**")
c1, c2, c3 = st.columns(3)
with c1:
    number_outpatient = st.number_input("Outpatient visits (prior year)", min_value=0, value=0)
with c2:
    number_emergency = st.number_input("Emergency visits (prior year)", min_value=0, value=1)
with c3:
    number_inpatient = st.number_input("Inpatient visits (prior year)", min_value=0, value=2)

st.markdown("**Labs & treatment**")
l1, l2, l3, l4 = st.columns(4)
with l1:
    max_glu_serum = st.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"])
with l2:
    A1Cresult = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"], index=3)
with l3:
    change = st.selectbox("Medication changed?", ["No", "Ch"])
with l4:
    diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])

with st.expander("💊 Individual medication statuses (optional — default 'No')"):
    med_values = {}
    med_cols = st.columns(4)
    for i, (key, label) in enumerate(MED_FIELDS):
        with med_cols[i % 4]:
            med_values[key] = st.selectbox(label, MED_OPTIONS, key=f"med_{key}")

st.divider()

# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------
if st.button("🔍 Predict readmission risk", type="primary", use_container_width=True):
    row = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "diag_1_category": diag_1_category,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "change": change,
        "diabetesMed": diabetesMed,
        **med_values,
        "total_prior_visits": number_outpatient + number_emergency + number_inpatient,
    }

    df = pd.DataFrame([row])
    df[categorical_cols] = df[categorical_cols].astype(str)
    df[categorical_cols] = encoder.transform(df[categorical_cols])
    df = df[feature_order]

    risk_prob = float(model.predict_proba(df)[0][1])
    risk_level = "High" if risk_prob > 0.5 else "Low"

    st.subheader("Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("30-day readmission risk", f"{risk_prob:.1%}")
        if risk_level == "High":
            st.error(f"**{risk_level} risk**")
        else:
            st.success(f"**{risk_level} risk**")
    with r2:
        st.progress(min(risk_prob, 1.0))
        st.caption(
            "Threshold: risk > 50% is flagged High. Because the model is tuned "
            "for recall, it flags more patients as high-risk than a "
            "precision-tuned model would — by design, to avoid missing "
            "genuinely at-risk patients."
        )

    with st.expander("See the exact feature row sent to the model"):
        st.dataframe(pd.DataFrame([row]))

st.divider()
st.caption(
    "Hospital Readmission Risk Prediction — MLOps Mini Project · "
    "Model + preprocessing pipeline from `src/`, served here via Streamlit "
    "using the same `models/best_model.pkl` and `models/preprocessor.pkl` "
    "artifacts the FastAPI service (`api/main.py`) uses."
)

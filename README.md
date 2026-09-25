# 🏥 Hospital Readmission Risk Prediction — MLOps Mini Project

An end-to-end **Machine Learning Operations (MLOps)** mini project that predicts the risk of a patient being readmitted to the hospital within 30 days.

The project demonstrates the complete ML lifecycle using **Git, GitHub, DVC, MLflow, FastAPI, Docker and Streamlit**.

---

## 🎯 Project Objective

The objective is to build a reproducible and deployable machine learning system for hospital readmission prediction.

The project covers:

* Source code version control using Git and GitHub
* Dataset versioning using DVC
* Reproducible ML pipelines using DVC DAG
* Experiment tracking using MLflow
* Model comparison and selection
* REST API deployment using FastAPI
* Interactive API testing through Swagger `/docs`
* Containerization using Docker
* Interactive prediction UI using Streamlit
* CI/retraining workflows using GitHub Actions

---

## 📊 Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

### Dataset details

* 101,766 hospital encounters
* 130 US hospitals
* 1999–2008
* 50 original features
* Target: hospital readmission within 30 days

The dataset is tracked using **DVC** rather than storing the full raw dataset directly in Git.

---

## 🧹 Data Preprocessing

The preprocessing pipeline performs:

* Missing-value handling
* Removal of unsuitable hospital discharge records
* Removal of duplicate patient encounters
* Removal of highly incomplete columns
* Removal of zero-variance features
* ICD-9 diagnosis categorization
* Feature engineering
* Categorical encoding
* Creation of the binary target variable

The target is:

```text
1 → Readmitted within 30 days
0 → Not readmitted within 30 days
```

---

## 🤖 Machine Learning Models

The project compares four classification models:

| Model               | Purpose                    |
| ------------------- | -------------------------- |
| Logistic Regression | Linear baseline classifier |
| Random Forest       | Ensemble tree model        |
| XGBoost             | Gradient boosting model    |
| LightGBM            | Gradient boosting model    |

Because the dataset is highly imbalanced, class balancing is applied during training.

### Evaluation metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC

Recall is used as the primary selection metric because missing a genuinely readmitted patient is an important error for this use case.

---

## 📈 Model Performance

The current experiment produced approximately:

| Model               | Accuracy | Precision |    Recall |    F1 |   AUC |
| ------------------- | -------: | --------: | --------: | ----: | ----: |
| Logistic Regression |    0.638 |     0.129 | **0.529** | 0.208 | 0.626 |
| Random Forest       |    0.761 |     0.149 |     0.352 | 0.209 | 0.630 |
| XGBoost             |    0.702 |     0.138 |     0.441 | 0.210 | 0.626 |
| LightGBM            |    0.720 |     0.146 |     0.437 | 0.219 | 0.627 |

The project selects the model using the recall-focused model-selection pipeline.

These values should not be interpreted as clinical-grade performance. This is an academic MLOps project demonstrating the engineering workflow.

---

# 🔄 MLOps Workflow

```text
Raw Dataset
     ↓
   DVC
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
DVC DAG
     ↓
Model Training
     ↓
    MLflow
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
 ┌───────────────┐
 │               │
FastAPI       Streamlit
 │               │
 └───────┬───────┘
         ↓
       Docker
         ↓
      Deployment
```

---

# 🗂️ Project Structure

```text
hospital-readmission-mlops/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   │   └── readmission_raw.csv.dvc
│   └── processed/
│
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── select_best_model.py
│   └── __init__.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retrain.yml
│
├── app.py
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── requirements-api.txt
├── .dvcignore
├── .gitignore
└── README.md
```

---

# 🧩 Technologies Used

| Technology     | Role                                     |
| -------------- | ---------------------------------------- |
| Git            | Version control                          |
| GitHub         | Remote repository                        |
| DVC            | Dataset and pipeline versioning          |
| DVC DAG        | Reproducible ML workflow                 |
| MLflow         | Experiment tracking and model management |
| FastAPI        | REST prediction API                      |
| Swagger Docs   | API testing                              |
| Streamlit      | Interactive prediction UI                |
| Docker         | Containerization                         |
| GitHub Actions | CI/retraining automation                 |
| Scikit-learn   | Machine learning                         |
| XGBoost        | Machine learning                         |
| LightGBM       | Machine learning                         |

---

# 🔬 DVC Pipeline

The DVC pipeline contains four main stages:

```text
preprocess
    ↓
train
    ↓
evaluate
    ↓
select_best_model
```

Run the complete pipeline using:

```bash
dvc repro
```

Check the pipeline DAG using:

```bash
dvc dag
```

Check the pipeline status using:

```bash
dvc status
```

---

# 📊 MLflow

Start MLflow locally:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

The experiment is:

```text
hospital-readmission-risk
```

MLflow records:

* Model parameters
* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC
* Model artifacts

---

# 🚀 FastAPI

Start the API:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Try the API using `/docs`

Open:

```text
http://127.0.0.1:8000/docs
```

Expand:

```text
POST /predict
```

Click:

```text
Try it out
```

Use the following example values.

### Basic patient information

```text
race: Caucasian
gender: Female
age: [70-80)
```

### Admission information

```text
admission_type_id: 1
discharge_disposition_id: 1
admission_source_id: 7
```

### Hospital stay

```text
time_in_hospital: 5
num_lab_procedures: 45
num_procedures: 1
num_medications: 18
number_outpatient: 0
number_emergency: 1
number_inpatient: 2
number_diagnoses: 9
```

### Diagnosis

```text
diag_1_category: Circulatory
```

Possible values:

```text
Circulatory
Respiratory
Digestive
Diabetes
Injury
Musculoskeletal
Genitourinary
Neoplasms
Other
Missing
```

### Laboratory values

```text
max_glu_serum: None
A1Cresult: >8
```

Possible values for `max_glu_serum`:

```text
None
Norm
>200
>300
```

Possible values for `A1Cresult`:

```text
None
Norm
>7
>8
```

### Medication information

For a simple test, keep the medication fields as:

```text
No
```

The supported medication status values are:

```text
No
Steady
Up
Down
```

### Treatment information

```text
change: Ch
diabetesMed: Yes
```

Possible `change` values:

```text
No
Ch
```

Possible `diabetesMed` values:

```text
Yes
No
```

The API returns:

```json
{
  "readmission_risk": 0.XXX,
  "risk_level": "High/Low",
  "model_version": "1"
}
```

---

# 🖥️ Streamlit Application

The project also includes an additional Streamlit interface.

Run:

```bash
streamlit run app.py
```

The application allows the user to:

1. Enter patient information
2. Enter hospital encounter information
3. Enter diagnosis and laboratory information
4. Enter medication information
5. Generate a readmission-risk prediction
6. View the predicted probability
7. View the predicted risk level

The Streamlit interface uses the same:

```text
models/best_model.pkl
models/preprocessor.pkl
```

used by the FastAPI service.

---

# 🐳 Docker

Build the Docker image:

```bash
docker build -t hospital-readmission-api .
```

Run the container:

```bash
docker run -p 8000:8000 hospital-readmission-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

The Docker container packages the FastAPI application, dependencies and model artifacts into a reproducible runtime environment.

---

# 🔁 GitHub Actions

The project contains GitHub Actions workflows for:

### CI

The CI workflow can:

```text
Checkout code
     ↓
Install dependencies
     ↓
DVC pull
     ↓
Lint
     ↓
DVC reproduce
     ↓
Evaluate model
     ↓
Apply recall gate
     ↓
Store model artifact
```

### Automatic Retraining

The retraining workflow can be triggered:

* Manually
* On a schedule
* When raw data changes

It retrains the candidate models and applies the performance gate before updating the model.

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd hospital-readmission-mlops
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### 1. Reproduce the ML pipeline

```bash
dvc repro
```

### 2. View the DVC DAG

```bash
dvc dag
```

### 3. Start MLflow

```bash
mlflow ui
```

### 4. Start FastAPI

```bash
uvicorn api.main:app --reload
```

### 5. Start Streamlit

In another terminal:

```bash
streamlit run app.py
```

### 6. Build Docker image

```bash
docker build -t hospital-readmission-api .
```

### 7. Run Docker

```bash
docker run -p 8000:8000 hospital-readmission-api
```

---

# ⚠️ Important Note About DVC

The raw dataset is intentionally not stored directly in Git.

Git stores:

```text
data/raw/readmission_raw.csv.dvc
```

The actual dataset should be stored in a configured DVC remote.

After cloning the repository, configure the DVC remote and run:

```bash
dvc pull
```

Then:

```bash
dvc repro
```

---

# 🎓 MLOps Concepts Demonstrated

This project demonstrates the following concepts:

```text
Git
 ↓
GitHub
 ↓
DVC Data Versioning
 ↓
DVC Pipeline / DAG
 ↓
MLflow Experiment Tracking
 ↓
Model Evaluation
 ↓
FastAPI REST API
 ↓
Swagger API Testing
 ↓
Streamlit UI
 ↓
Docker Containerization
 ↓
GitHub Actions Automation
```

---

## 👨‍🎓 Academic Project

**Subject:** MLOps
**Project:** Hospital Readmission Risk Prediction
**Type:** Mini Project

This project is intended for academic demonstration of an end-to-end MLOps workflow and is not intended for clinical decision-making.

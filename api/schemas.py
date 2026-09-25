from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MedStatus = Literal["No", "Steady", "Up", "Down"]
ChangeStatus = Literal["No", "Ch"]
YesNo = Literal["No", "Yes"]
DiagCategory = Literal[
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]


class PatientData(BaseModel):
    # Demographics
    race: str = Field(..., examples=["Caucasian"])
    gender: Literal["Male", "Female"] = Field(..., examples=["Female"])
    age: str = Field(..., examples=["[70-80)"])

    # Admission / discharge (raw UCI numeric IDs — see IDs_mapping.csv from
    # the original dataset for what each code means)
    admission_type_id: int = Field(..., ge=1, le=8, examples=[1])
    discharge_disposition_id: int = Field(..., ge=1, le=28, examples=[1])
    admission_source_id: int = Field(..., ge=1, le=25, examples=[7])

    # Encounter details
    time_in_hospital: int = Field(..., ge=1, le=14, examples=[5])
    num_lab_procedures: int = Field(..., ge=0, examples=[45])
    num_procedures: int = Field(..., ge=0, examples=[1])
    num_medications: int = Field(..., ge=0, examples=[18])
    number_outpatient: int = Field(..., ge=0, examples=[0])
    number_emergency: int = Field(..., ge=0, examples=[1])
    number_inpatient: int = Field(..., ge=0, examples=[2])
    number_diagnoses: int = Field(..., ge=1, examples=[9])

    # Primary diagnosis, pre-bucketed into a clinical category (see
    # src/preprocessing.py::bucket_icd9 for the raw ICD-9 -> category mapping)
    diag_1_category: DiagCategory = Field(..., examples=["Circulatory"])

    # Labs
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = Field(..., examples=["None"])
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = Field(..., examples=[">8"])

    # Medications
    metformin: MedStatus = "No"
    repaglinide: MedStatus = "No"
    nateglinide: MedStatus = "No"
    chlorpropamide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    acetohexamide: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    tolbutamide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    acarbose: MedStatus = "No"
    miglitol: MedStatus = "No"
    troglitazone: MedStatus = "No"
    tolazamide: MedStatus = "No"
    insulin: MedStatus = "No"
    glyburide_metformin: MedStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: MedStatus = Field("No", alias="glipizide-metformin")
    metformin_rosiglitazone: MedStatus = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedStatus = Field("No", alias="metformin-pioglitazone")

    change: ChangeStatus = Field(..., examples=["Ch"])
    diabetesMed: YesNo = Field(..., examples=["Yes"])

    model_config = ConfigDict(populate_by_name=True)


class PredictionResponse(BaseModel):
    readmission_risk: float
    risk_level: str
    model_version: str

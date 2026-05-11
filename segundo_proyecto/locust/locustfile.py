import random
from locust import HttpUser, task, between

SAMPLE_PAYLOADS = [
    {
        "time_in_hospital": 3, "num_lab_procedures": 41, "num_procedures": 0,
        "num_medications": 13, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "number_diagnoses": 9,
        "age_encoded": 65, "gender_encoded": 1, "race_encoded": 0,
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
        "a1cresult_encoded": 0, "max_glu_serum_encoded": 0,
        "metformin_encoded": 1, "repaglinide_encoded": 0, "nateglinide_encoded": 0,
        "chlorpropamide_encoded": 0, "glimepiride_encoded": 0, "acetohexamide_encoded": 0,
        "glipizide_encoded": 0, "glyburide_encoded": 0, "tolbutamide_encoded": 0,
        "pioglitazone_encoded": 0, "rosiglitazone_encoded": 0, "acarbose_encoded": 0,
        "miglitol_encoded": 0, "troglitazone_encoded": 0, "tolazamide_encoded": 0,
        "examide_encoded": 0, "citoglipton_encoded": 0, "insulin_encoded": 1,
        "glyburide_metformin_encoded": 0, "glipizide_metformin_encoded": 0,
        "glimepiride_pioglitazone_encoded": 0, "metformin_rosiglitazone_encoded": 0,
        "metformin_pioglitazone_encoded": 0,
        "change_encoded": 1, "diabetesmed_encoded": 1,
        "diag_1_code": 250.0, "diag_2_code": 401.0, "diag_3_code": 276.0,
    },
    {
        "time_in_hospital": 7, "num_lab_procedures": 58, "num_procedures": 2,
        "num_medications": 20, "number_outpatient": 1, "number_emergency": 0,
        "number_inpatient": 2, "number_diagnoses": 7,
        "age_encoded": 75, "gender_encoded": 0, "race_encoded": 1,
        "admission_type_id": 2, "discharge_disposition_id": 3, "admission_source_id": 1,
        "a1cresult_encoded": 2, "max_glu_serum_encoded": 1,
        "metformin_encoded": 0, "repaglinide_encoded": 0, "nateglinide_encoded": 0,
        "chlorpropamide_encoded": 0, "glimepiride_encoded": 1, "acetohexamide_encoded": 0,
        "glipizide_encoded": 2, "glyburide_encoded": 0, "tolbutamide_encoded": 0,
        "pioglitazone_encoded": 0, "rosiglitazone_encoded": 0, "acarbose_encoded": 0,
        "miglitol_encoded": 0, "troglitazone_encoded": 0, "tolazamide_encoded": 0,
        "examide_encoded": 0, "citoglipton_encoded": 0, "insulin_encoded": 2,
        "glyburide_metformin_encoded": 0, "glipizide_metformin_encoded": 0,
        "glimepiride_pioglitazone_encoded": 0, "metformin_rosiglitazone_encoded": 0,
        "metformin_pioglitazone_encoded": 0,
        "change_encoded": 1, "diabetesmed_encoded": 1,
        "diag_1_code": 250.0, "diag_2_code": 428.0, "diag_3_code": 585.0,
    },
    {
        "time_in_hospital": 1, "num_lab_procedures": 25, "num_procedures": 1,
        "num_medications": 8, "number_outpatient": 0, "number_emergency": 1,
        "number_inpatient": 0, "number_diagnoses": 5,
        "age_encoded": 45, "gender_encoded": 1, "race_encoded": 2,
        "admission_type_id": 3, "discharge_disposition_id": 1, "admission_source_id": 7,
        "a1cresult_encoded": 1, "max_glu_serum_encoded": 0,
        "metformin_encoded": 0, "repaglinide_encoded": 0, "nateglinide_encoded": 0,
        "chlorpropamide_encoded": 0, "glimepiride_encoded": 0, "acetohexamide_encoded": 0,
        "glipizide_encoded": 0, "glyburide_encoded": 1, "tolbutamide_encoded": 0,
        "pioglitazone_encoded": 0, "rosiglitazone_encoded": 0, "acarbose_encoded": 0,
        "miglitol_encoded": 0, "troglitazone_encoded": 0, "tolazamide_encoded": 0,
        "examide_encoded": 0, "citoglipton_encoded": 0, "insulin_encoded": 0,
        "glyburide_metformin_encoded": 0, "glipizide_metformin_encoded": 0,
        "glimepiride_pioglitazone_encoded": 0, "metformin_rosiglitazone_encoded": 0,
        "metformin_pioglitazone_encoded": 0,
        "change_encoded": 0, "diabetesmed_encoded": 1,
        "diag_1_code": 250.0, "diag_2_code": 272.0, "diag_3_code": 401.0,
    },
]


class DiabetesAPIUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(8)
    def predict(self):
        payload = random.choice(SAMPLE_PAYLOADS)
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def model_info(self):
        self.client.get("/model-info")

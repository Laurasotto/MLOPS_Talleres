import random
from locust import HttpUser, task, between

# Aquí definimos tres perfiles de pacientes distintos que usamos en el test de carga.
# Locust elige uno al azar en cada request para simular tráfico más realista
# en lugar de mandar siempre el mismo payload.
#
# Los valores corresponden a las features codificadas que espera la API:
# - age_encoded: punto medio del rango de edad ([60-70) → 65)
# - gender_encoded: 0=Masculino, 1=Femenino
# - race_encoded: 0=Caucásico, 1=Afroamericano, 2=Hispano, etc.
# - medicamentos: 0=No, 1=Estable, 2=Dosis aumentada, 3=Dosis reducida
SAMPLE_PAYLOADS = [
    {
        # Perfil 1: paciente de 65 años, femenina, caucásica, con metformina e insulina estables
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
        # Perfil 2: paciente de 75 años, masculino, afroamericano, con más complicaciones
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
        # Perfil 3: paciente de 45 años, femenina, hispana, hospitalización corta
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
    # Cada usuario virtual espera entre 0.5 y 2 segundos entre requests.
    # Esto simula un comportamiento más realista que mandar requests sin pausa.
    wait_time = between(0.5, 2.0)

    @task(8)
    def predict(self):
        # Esta tarea tiene peso 8 — significa que por cada 10 requests que hace un usuario,
        # 8 son predicciones y 2 son chequeos de salud/info del modelo.
        # Elegimos un payload al azar de los tres perfiles definidos arriba.
        payload = random.choice(SAMPLE_PAYLOADS)
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,  # catch_response=True nos permite marcar manualmente éxito o fallo
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Model not loaded")  # la API no tiene modelo cargado todavía
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def health_check(self):
        # Chequeo de salud — verifica que el servidor responde.
        # Peso 1: se ejecuta una vez por cada 10 requests totales.
        self.client.get("/health")

    @task(1)
    def model_info(self):
        # Consulta la información del modelo cargado — qué versión y alias está activo.
        # Peso 1: se ejecuta una vez por cada 10 requests totales.
        self.client.get("/model-info")

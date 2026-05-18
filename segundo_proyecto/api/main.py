# ─────────────────────────────────────────────────────────────────────────────
# Aquí importamos todo lo que necesita la API para funcionar.
# ─────────────────────────────────────────────────────────────────────────────

# Librerías estándar de Python
import os       # para leer variables de entorno
import time     # para medir cuánto tarda cada predicción
import uuid     # para generar un ID único por cada request
import logging  # para registrar mensajes en los logs del servidor
from contextlib import asynccontextmanager  # para definir qué hacer al arrancar y apagar la app
from typing import Optional

# MLflow nos permite cargar el modelo champion directamente desde el registry
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# numpy para armar el arreglo de features que recibe el modelo
import numpy as np

# FastAPI es el framework que usamos para construir la API REST
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

# Pydantic nos permite definir y validar el formato de los datos de entrada y salida
from pydantic import BaseModel, Field

# prometheus_client nos permite exponer métricas de uso que Prometheus puede recolectar
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# SQLAlchemy para conectarnos a PostgreSQL y guardar los logs de inferencia
from sqlalchemy import create_engine, text

log = logging.getLogger("uvicorn.error")

# ─────────────────────────────────────────────────────────────────────────────
# Configuración general de la API.
# Usamos variables de entorno para no hardcodear URLs ni credenciales en el código.
# Si la variable no existe, usamos el valor por defecto que funciona dentro de Docker.
# ─────────────────────────────────────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes_readmission_model")

# MODEL_ALIAS define qué versión del modelo cargamos — siempre cargamos el 'champion',
# que es el que el pipeline promovió como el mejor hasta ahora.
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@postgres-service:5432/mlops_db"
)

# ─────────────────────────────────────────────────────────────────────────────
# Métricas de Prometheus
# Aquí definimos qué cosas queremos medir y exponer para monitoreo.
# Prometheus va a venir a buscar estos datos al endpoint /metrics.
# ─────────────────────────────────────────────────────────────────────────────

# Counter: cuenta el total de requests de predicción, separados por si fueron exitosos o fallidos.
# Nos sirve para saber cuántas predicciones se hacen por minuto y cuántas fallan.
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["status"]
)

# Histogram: mide el tiempo que tarda cada predicción en segundos.
# Los buckets definen los rangos — por ejemplo cuántas predicciones tardaron menos de 0.1s, menos de 0.5s, etc.
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# ─────────────────────────────────────────────────────────────────────────────
# Caché del modelo en memoria
# Usamos un diccionario para guardar el modelo cargado y sus metadatos.
# Esto lo hacemos para no ir a buscar el modelo a MLflow en cada request —
# solo lo cargamos una vez al arrancar la app y lo mantenemos en memoria.
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: dict = {}


def load_champion_model():
    # Aquí nos conectamos a MLflow y cargamos el modelo que tiene el alias 'champion'.
    # Al usar el alias en lugar de un número de versión fijo, la API siempre carga
    # automáticamente la mejor versión disponible sin necesidad de cambiar el código.
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    # Guardamos el modelo y sus metadatos en el caché para no tener que cargarlo de nuevo
    _model_cache["model"] = model
    _model_cache["version"] = mv.version
    _model_cache["run_id"] = mv.run_id
    _model_cache["name"] = MODEL_NAME
    _model_cache["alias"] = MODEL_ALIAS
    log.info("Loaded champion model version %s (run_id=%s)", mv.version, mv.run_id)
    return model


def get_model():
    # Si el modelo ya está en caché lo devolvemos directamente.
    # Si no (por ejemplo si la API arrancó antes de que el pipeline entrenara el primer modelo),
    # intentamos cargarlo en este momento — carga lazy.
    if "model" not in _model_cache:
        load_champion_model()
    return _model_cache["model"]


# ─────────────────────────────────────────────────────────────────────────────
# Conexión a la base de datos (inicialización lazy)
# Creamos el motor de conexión solo cuando lo necesitamos por primera vez,
# no al arrancar la app — esto evita errores si la BD todavía no está lista.
# ─────────────────────────────────────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URI)
    return _engine


def ensure_inference_table():
    # Aquí creamos la tabla donde vamos a guardar cada predicción que haga la API.
    # Guardamos la entrada, el resultado, la probabilidad, el modelo usado y el tiempo de respuesta —
    # esto nos permite auditar qué predijo el modelo y cuándo.
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS inference_logs"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS inference_logs.predictions (
                id                  SERIAL PRIMARY KEY,
                request_id          VARCHAR(36),
                inference_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                input_data          JSONB,
                prediction          INT,
                prediction_label    VARCHAR(50),
                probability         FLOAT,
                model_name          VARCHAR(255),
                model_version       VARCHAR(50),
                model_alias         VARCHAR(50),
                response_time_ms    FLOAT,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — acciones al arrancar y apagar la app
# Esta función define qué hacer cuando FastAPI inicia y cuando se cierra.
# El yield separa el código de arranque (antes) del de cierre (después).
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al arrancar, intentamos pre-cargar el modelo en memoria para que el primer request
    # no tenga que esperar a que se descargue de MLflow.
    try:
        load_champion_model()
    except Exception as e:
        log.warning("Could not pre-load model at startup: %s. Will retry on first request.", e)
    # También creamos la tabla de logs si no existe todavía
    try:
        ensure_inference_table()
    except Exception as e:
        log.warning("Could not ensure inference table: %s", e)
    yield
    # Aquí podríamos agregar limpieza al apagar la app (cerrar conexiones, etc.)


# Creamos la aplicación FastAPI con su documentación automática
app = FastAPI(
    title="Diabetes Readmission API",
    description="MLOps Proyecto 2 — inference API backed by MLflow champion model",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# Schemas de entrada y salida
# Pydantic valida automáticamente que los datos recibidos tengan el tipo correcto.
# Si falta un campo obligatorio o tiene el tipo equivocado, FastAPI devuelve un error 422.
# ─────────────────────────────────────────────────────────────────────────────

# PredictRequest define exactamente qué campos debe enviar quien llame a la API.
# Los campos con Field(0) tienen valor por defecto 0 — son opcionales en el request.
# Los campos con Field(...) son obligatorios — el ... es la forma de Pydantic de decir "requerido".
class PredictRequest(BaseModel):
    time_in_hospital: float = Field(..., example=3)
    num_lab_procedures: float = Field(..., example=41)
    num_procedures: float = Field(..., example=0)
    num_medications: float = Field(..., example=13)
    number_outpatient: float = Field(..., example=0)
    number_emergency: float = Field(..., example=0)
    number_inpatient: float = Field(..., example=0)
    number_diagnoses: float = Field(..., example=9)
    age_encoded: float = Field(..., example=65)
    gender_encoded: float = Field(..., example=1, description="0=Male, 1=Female")
    race_encoded: float = Field(..., example=0, description="0=Caucasian…4=Other")
    admission_type_id: float = Field(..., example=1)
    discharge_disposition_id: float = Field(..., example=1)
    admission_source_id: float = Field(..., example=7)
    a1cresult_encoded: float = Field(0, example=0)
    max_glu_serum_encoded: float = Field(0, example=0)
    metformin_encoded: float = Field(0, example=1)
    repaglinide_encoded: float = Field(0)
    nateglinide_encoded: float = Field(0)
    chlorpropamide_encoded: float = Field(0)
    glimepiride_encoded: float = Field(0)
    acetohexamide_encoded: float = Field(0)
    glipizide_encoded: float = Field(0)
    glyburide_encoded: float = Field(0)
    tolbutamide_encoded: float = Field(0)
    pioglitazone_encoded: float = Field(0)
    rosiglitazone_encoded: float = Field(0)
    acarbose_encoded: float = Field(0)
    miglitol_encoded: float = Field(0)
    troglitazone_encoded: float = Field(0)
    tolazamide_encoded: float = Field(0)
    examide_encoded: float = Field(0)
    citoglipton_encoded: float = Field(0)
    insulin_encoded: float = Field(0, example=1)
    glyburide_metformin_encoded: float = Field(0)
    glipizide_metformin_encoded: float = Field(0)
    glimepiride_pioglitazone_encoded: float = Field(0)
    metformin_rosiglitazone_encoded: float = Field(0)
    metformin_pioglitazone_encoded: float = Field(0)
    change_encoded: float = Field(0, example=1)
    diabetesmed_encoded: float = Field(1, example=1)
    diag_1_code: float = Field(0, example=250)
    diag_2_code: float = Field(0, example=401)
    diag_3_code: float = Field(0, example=276)


# PredictResponse define qué devuelve la API después de hacer una predicción
class PredictResponse(BaseModel):
    request_id: str        # ID único de este request, para poder rastrearlo en los logs
    prediction: int        # 1 = readmitido en <30 días, 0 = no readmitido en <30 días
    prediction_label: str  # etiqueta legible: "readmitted_early" o "not_readmitted_early"
    probability: float     # probabilidad de la clase predicha (entre 0 y 1)
    model_name: str        # nombre del modelo en el registry de MLflow
    model_version: str     # versión exacta del modelo que hizo la predicción
    model_alias: str       # alias usado (siempre "champion")
    response_time_ms: float  # cuántos milisegundos tardó en responder


# ModelInfoResponse define qué devuelve el endpoint /model-info
class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str
    run_id: str    # ID del run de MLflow que generó este modelo
    status: str


# ─────────────────────────────────────────────────────────────────────────────
# Orden de las features
# Este orden debe ser exactamente el mismo que se usó al entrenar el modelo.
# Si los features llegan en un orden diferente, las predicciones serían incorrectas
# porque el modelo aprendió que "columna 0 = time_in_hospital", etc.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "age_encoded", "gender_encoded", "race_encoded",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "a1cresult_encoded", "max_glu_serum_encoded",
    "metformin_encoded", "repaglinide_encoded", "nateglinide_encoded",
    "chlorpropamide_encoded", "glimepiride_encoded", "acetohexamide_encoded",
    "glipizide_encoded", "glyburide_encoded", "tolbutamide_encoded",
    "pioglitazone_encoded", "rosiglitazone_encoded", "acarbose_encoded",
    "miglitol_encoded", "troglitazone_encoded", "tolazamide_encoded",
    "examide_encoded", "citoglipton_encoded", "insulin_encoded",
    "glyburide_metformin_encoded", "glipizide_metformin_encoded",
    "glimepiride_pioglitazone_encoded", "metformin_rosiglitazone_encoded",
    "metformin_pioglitazone_encoded",
    "change_encoded", "diabetesmed_encoded",
    "diag_1_code", "diag_2_code", "diag_3_code",
]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints de la API
# ─────────────────────────────────────────────────────────────────────────────

# Endpoint de salud — solo devuelve {"status": "ok"}.
# Se usa para que Docker o Kubernetes sepan que el contenedor está vivo.
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint de información del modelo — muestra qué versión del modelo está cargada actualmente.
# Útil para verificar que la API esté usando el champion correcto después de un nuevo entrenamiento.
@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if "model" not in _model_cache:
        try:
            load_champion_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    return ModelInfoResponse(
        model_name=_model_cache["name"],
        model_version=_model_cache["version"],
        model_alias=_model_cache["alias"],
        run_id=_model_cache["run_id"],
        status="loaded",
    )


# Endpoint principal — recibe los datos de un paciente y devuelve si va a ser readmitido en <30 días.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Empezamos a medir el tiempo desde que llega el request
    start = time.time()

    # Generamos un ID único para este request — nos permite rastrearlo en los logs de la BD
    request_id = str(uuid.uuid4())

    # Obtenemos el modelo del caché (si no está cargado, lo intenta cargar aquí)
    try:
        model = get_model()
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    # Armamos el arreglo de features en el orden exacto que espera el modelo.
    # getattr(request, col) extrae el valor de cada campo del request por nombre.
    features = np.array([[getattr(request, col) for col in FEATURE_ORDER]])

    # Hacemos la predicción y calculamos la probabilidad de la clase predicha
    try:
        prediction = int(model.predict(features)[0])
        # predict_proba devuelve probabilidades para cada clase — tomamos la de la clase predicha
        probability = float(model.predict_proba(features)[0][prediction])
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Calculamos cuánto tardó la predicción y lo registramos en las métricas de Prometheus
    elapsed_ms = (time.time() - start) * 1000
    REQUEST_LATENCY.observe(elapsed_ms / 1000)
    REQUEST_COUNT.labels(status="success").inc()

    # Convertimos la predicción numérica (0 o 1) a una etiqueta legible
    label = "readmitted_early" if prediction == 1 else "not_readmitted_early"

    # Guardamos este request en la BD de logs — lo hacemos en modo "best-effort":
    # si falla (por ejemplo si la BD no está disponible), solo logueamos el warning
    # y seguimos — no queremos que un fallo en los logs arruine la respuesta al usuario.
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO inference_logs.predictions
                        (request_id, input_data, prediction, prediction_label,
                         probability, model_name, model_version, model_alias,
                         response_time_ms)
                    VALUES
                        (:request_id, :input_data::jsonb, :prediction, :label,
                         :probability, :model_name, :model_version, :model_alias,
                         :response_time_ms)
                """),
                {
                    "request_id": request_id,
                    "input_data": request.model_dump_json(),
                    "prediction": prediction,
                    "label": label,
                    "probability": probability,
                    "model_name": _model_cache.get("name", MODEL_NAME),
                    "model_version": _model_cache.get("version", "unknown"),
                    "model_alias": MODEL_ALIAS,
                    "response_time_ms": elapsed_ms,
                },
            )
            conn.commit()
    except Exception as e:
        log.warning("Failed to log inference to DB: %s", e)

    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        prediction_label=label,
        probability=probability,
        model_name=_model_cache.get("name", MODEL_NAME),
        model_version=_model_cache.get("version", "unknown"),
        model_alias=MODEL_ALIAS,
        response_time_ms=elapsed_ms,
    )


# Endpoint para recargar el modelo manualmente — útil cuando el pipeline entrena
# un nuevo champion y queremos que la API lo cargue sin tener que reiniciar el contenedor.
@app.post("/reload-model")
def reload_model():
    try:
        load_champion_model()
        return {"status": "reloaded", "version": _model_cache["version"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# Endpoint de métricas — Prometheus llama a este endpoint periódicamente
# para recolectar los contadores y histogramas que definimos arriba.
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

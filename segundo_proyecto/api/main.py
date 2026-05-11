import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import create_engine, text

log = logging.getLogger("uvicorn.error")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes_readmission_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@postgres-service:5432/mlops_db"
)

# ──────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# ──────────────────────────────────────────────
# Model state (loaded once at startup, reloaded on demand)
# ──────────────────────────────────────────────
_model_cache: dict = {}


def load_champion_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    _model_cache["model"] = model
    _model_cache["version"] = mv.version
    _model_cache["run_id"] = mv.run_id
    _model_cache["name"] = MODEL_NAME
    _model_cache["alias"] = MODEL_ALIAS
    log.info("Loaded champion model version %s (run_id=%s)", mv.version, mv.run_id)
    return model


def get_model():
    if "model" not in _model_cache:
        load_champion_model()
    return _model_cache["model"]


# ──────────────────────────────────────────────
# Database engine (lazy init)
# ──────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URI)
    return _engine


def ensure_inference_table():
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


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_champion_model()
    except Exception as e:
        log.warning("Could not pre-load model at startup: %s. Will retry on first request.", e)
    try:
        ensure_inference_table()
    except Exception as e:
        log.warning("Could not ensure inference table: %s", e)
    yield


app = FastAPI(
    title="Diabetes Readmission API",
    description="MLOps Proyecto 2 — inference API backed by MLflow champion model",
    version="1.0.0",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
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


class PredictResponse(BaseModel):
    request_id: str
    prediction: int
    prediction_label: str
    probability: float
    model_name: str
    model_version: str
    model_alias: str
    response_time_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str
    run_id: str
    status: str


# ──────────────────────────────────────────────
# Feature ordering (must match training)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    request_id = str(uuid.uuid4())

    try:
        model = get_model()
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    features = np.array([[getattr(request, col) for col in FEATURE_ORDER]])

    try:
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][prediction])
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    elapsed_ms = (time.time() - start) * 1000
    REQUEST_LATENCY.observe(elapsed_ms / 1000)
    REQUEST_COUNT.labels(status="success").inc()

    label = "readmitted_early" if prediction == 1 else "not_readmitted_early"

    # Log to database (best-effort)
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


@app.post("/reload-model")
def reload_model():
    try:
        load_champion_model()
        return {"status": "reloaded", "version": _model_cache["version"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

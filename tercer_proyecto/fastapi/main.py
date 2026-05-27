# API de inferencia para predecir el precio de propiedades inmobiliarias.
# Carga el modelo que esté marcado como "production" en el MLflow Registry
# y lo sirve a través de un endpoint REST. Cada predicción queda registrada
# en la tabla raw.inference_events de PostgreSQL para trazabilidad.
# También expone métricas en formato Prometheus para que Grafana las grafique.

import os
import time
import logging
import threading
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

# Dirección del servidor de MLflow dentro del cluster de Kubernetes.
# Si no se define la variable de entorno se usa la dirección por defecto del cluster.
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow.svc.cluster.local:5000")
MODEL_NAME = "precio_propiedades_grupo4"
PROD_ALIAS = "production"

# Configuración de la base de datos. Los valores vienen del Secret de Kubernetes.
DB_HOST = os.environ.get("DB_HOST", "postgres-service.postgres.svc.cluster.local")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "mlops")
DB_USER = os.environ.get("DB_USER", "grupo4")
DB_PASS = os.environ.get("DB_PASS", "grupo4pass")

# Contador de peticiones agrupado por resultado (ok, error, no_model).
REQUEST_COUNT   = Counter("predict_requests_total", "Total de solicitudes de prediccion", ["status"])
# Histograma de latencia: mide cuánto tarda cada predicción en segundos.
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Latencia de prediccion en segundos")

# El modelo y su metadata se guardan en variables globales protegidas con un lock
# porque FastAPI maneja múltiples peticiones en paralelo y no queremos que una
# recarga de modelo a mitad de una predicción cause problemas de concurrencia.
_model      = None
_model_meta = {}
_model_lock = threading.Lock()


def _db_conn():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)


def _load_model():
    """Descarga el modelo productivo desde MLflow y lo deja listo en memoria."""
    global _model, _model_meta
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    try:
        mv     = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
        loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{PROD_ALIAS}")
        with _model_lock:
            _model      = loaded
            _model_meta = {
                "model_name": MODEL_NAME,
                "version":    mv.version,
                "alias":      PROD_ALIAS,
                "run_id":     mv.run_id,
            }
        logging.info(f"Modelo cargado: {MODEL_NAME} v{mv.version}")
    except Exception as exc:
        # Si no hay ningún modelo con alias "production" en el registry la API
        # arranca igual pero responde 503 en /predict hasta que Airflow promueva uno.
        logging.warning(f"No se pudo cargar el modelo de produccion: {exc}")
        with _model_lock:
            _model      = None
            _model_meta = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta una sola vez al arrancar el servidor, antes de recibir tráfico.
    _load_model()
    yield


app = FastAPI(
    title="API de Inferencia — Precio de Propiedades",
    description="Grupo 4 — Thomas Rivera & Laura Sotto",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    bed:        float   # número de habitaciones
    bath:       float   # número de baños
    acre_lot:   float   # tamaño del terreno en acres
    house_size: float   # área habitable en pies cuadrados
    status:     str     # "for_sale" o "sold"
    city:       str
    state:      str
    zip_code:   str


class PredictResponse(BaseModel):
    predicted_price: float
    model_version:   str
    model_name:      str


def _log_event(input_data: dict, prediction: float | None, status: str, error: str | None, version: str | None):
    """Guarda el resultado de cada predicción en la base de datos para auditoría."""
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            import json
            cur.execute(
                """
                INSERT INTO raw.inference_events
                    (input_data, prediction, model_version, model_alias, response_status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (json.dumps(input_data), prediction, version, PROD_ALIAS, status, error),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        # Si falla el registro en base de datos no queremos que la predicción falle también.
        # Solo logueamos la advertencia y seguimos.
        logging.warning(f"No se pudo registrar el evento de inferencia: {exc}")


@app.get("/health")
def health():
    """Indica si el servicio está corriendo y si tiene un modelo cargado."""
    with _model_lock:
        loaded = _model is not None
        meta   = dict(_model_meta)
    return {"status": "ok" if loaded else "no_model", "model_loaded": loaded, **meta}


@app.get("/model-info")
def model_info():
    """Devuelve la versión y el run_id del modelo que está sirviendo en este momento."""
    with _model_lock:
        meta = dict(_model_meta)
    if not meta:
        raise HTTPException(status_code=503, detail="No hay modelo con alias 'production' en MLflow")
    return meta


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Recibe las características de una propiedad y devuelve el precio estimado."""
    with _model_lock:
        current_model   = _model
        current_version = _model_meta.get("version")

    if current_model is None:
        REQUEST_COUNT.labels(status="no_model").inc()
        raise HTTPException(
            status_code=503,
            detail="No hay modelo en produccion. Espera a que Airflow entrene y promueva uno.",
        )

    input_dict = req.model_dump()
    df = pd.DataFrame([input_dict])

    t0 = time.time()
    try:
        prediction = float(current_model.predict(df)[0])
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        _log_event(input_dict, None, "error", str(exc), current_version)
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {exc}")

    REQUEST_LATENCY.observe(time.time() - t0)
    REQUEST_COUNT.labels(status="ok").inc()
    _log_event(input_dict, prediction, "ok", None, current_version)

    return PredictResponse(
        predicted_price=prediction,
        model_version=str(current_version),
        model_name=MODEL_NAME,
    )


@app.post("/reload-model")
def reload_model():
    """Fuerza la recarga del modelo desde MLflow sin necesidad de reiniciar el pod.
    Útil cuando Airflow promueve un modelo nuevo y queremos que la API lo use de inmediato."""
    _load_model()
    with _model_lock:
        if _model is None:
            raise HTTPException(status_code=503, detail="No hay modelo con alias 'production' en MLflow")
        return {"message": "Modelo recargado exitosamente", **_model_meta}


@app.get("/metrics")
def metrics():
    """Expone las métricas en formato Prometheus para que Grafana las consuma."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

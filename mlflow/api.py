"""
API de inferencia — carga el modelo desde MLflow Model Registry
Uso:
    pip install fastapi uvicorn mlflow boto3
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── Configuración ─────────────────────────────────────────────────────────────
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
os.environ['AWS_ACCESS_KEY_ID']      = os.getenv('AWS_ACCESS_KEY_ID', 'admin')
os.environ['AWS_SECRET_ACCESS_KEY']  = os.getenv('AWS_SECRET_ACCESS_KEY', 'supersecret')

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME          = 'diabetes_rf_regressor'
MODEL_STAGE         = 'latest'  # puede ser 'Production', '1', etc.

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Cargar modelo al iniciar la API ───────────────────────────────────────────
print(f"Cargando modelo '{MODEL_NAME}' desde MLflow...")
try:
    model = mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{MODEL_STAGE}')
    print("Modelo cargado exitosamente ✓")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    model = None

# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="API de Inferencia — Diabetes",
    description="Predice la progresión de diabetes usando el modelo registrado en MLflow",
    version="1.0.0"
)

# ── Esquema de entrada ────────────────────────────────────────────────────────
# El dataset de diabetes tiene 10 features:
# age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
class PredictionRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    class Config:
        json_schema_extra = {
            "example": {
                "age":  0.038,
                "sex":  0.050,
                "bmi":  0.061,
                "bp":  -0.025,
                "s1":  -0.044,
                "s2":  -0.034,
                "s3":  -0.043,
                "s4":  -0.002,
                "s5":   0.019,
                "s6":  -0.017
            }
        }

class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_stage: str

class BatchRequest(BaseModel):
    instances: List[PredictionRequest]

class BatchResponse(BaseModel):
    predictions: List[float]
    model_name: str
    model_stage: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "API de inferencia MLflow activa", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    features = np.array([[
        request.age, request.sex, request.bmi, request.bp,
        request.s1, request.s2, request.s3, request.s4,
        request.s5, request.s6
    ]])
    
    prediction = model.predict(features)[0]
    
    return PredictionResponse(
        prediction=float(prediction),
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE
    )

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    features = np.array([[
        r.age, r.sex, r.bmi, r.bp,
        r.s1, r.s2, r.s3, r.s4, r.s5, r.s6
    ] for r in request.instances])
    
    predictions = model.predict(features).tolist()
    
    return BatchResponse(
        predictions=predictions,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE
    )

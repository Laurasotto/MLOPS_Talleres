import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.join("model", "dt_penguins.joblib")
COLS_PATH  = os.path.join("model", "model_columns.joblib")

app = FastAPI(title="Penguins API", version="1.0.0")

class PenguinIn(BaseModel):
    bill_length_mm: float | None = None
    bill_depth_mm: float | None = None
    flipper_length_mm: float | None = None
    body_mass_g: float | None = None
    sex: str | None = None
    island: str | None = None
    year: int | None = None

@app.on_event("startup")
def load_artifacts():
    global model, model_cols
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
        raise RuntimeError("Falta el modelo. Ejecuta primero: python train.py")
    model = joblib.load(MODEL_PATH)
    model_cols = joblib.load(COLS_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PenguinIn):
    try:
        X = pd.DataFrame([payload.model_dump()])
        X = pd.get_dummies(X, drop_first=True)

        # Alinear columnas con las usadas en entrenamiento
        X = X.reindex(columns=model_cols, fill_value=0)

        # Nulos numéricos (por si llegan)
        X = X.fillna(X.median(numeric_only=True))

        pred = model.predict(X)[0]
        return {"prediction": str(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


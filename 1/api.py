import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

MODEL_DIR = os.getenv("MODEL_DIR", "model")
COLS_PATH   = os.path.join(MODEL_DIR, "model_columns.joblib")
ACTIVE_PATH = os.path.join(MODEL_DIR, "active_model.txt")

app = FastAPI(title="Penguins API", version="1.0.0")

# --------- Schemas ---------
class PenguinIn(BaseModel):
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[str] = None
    island: Optional[str] = None
    year: Optional[int] = None

class SelectModelIn(BaseModel):
    name: str

# --------- Helpers ---------
def list_models() -> List[str]:
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted([
        f[:-6] for f in os.listdir(MODEL_DIR)
        if f.endswith(".joblib") and f != "model_columns.joblib"
    ])

def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def get_active_name() -> str:
    if os.path.exists(ACTIVE_PATH):
        name = open(ACTIVE_PATH, "r").read().strip()
        if name:
            return name
    models = list_models()
    return models[0] if models else ""

def set_active_name(name: str) -> None:
    # valida que exista
    _ = load_model(name)
    with open(ACTIVE_PATH, "w") as f:
        f.write(name)

# --------- Globals ---------
model = None
model_cols = None
active_name = None

@app.on_event("startup")
def load_artifacts():
    global model, model_cols, active_name

    if not os.path.exists(COLS_PATH):
        raise RuntimeError("Faltan columnas. Ejecuta primero: python3 modelo1.py")

    model_cols = joblib.load(COLS_PATH)

    active_name = get_active_name()
    if not active_name:
        raise RuntimeError("No hay modelos en /model. Ejecuta primero: python3 modelo1.py")

    model = load_model(active_name)

# --------- Endpoints ---------
@app.get("/health")
def health():
    return {"status": "ok", "active_model": active_name}

@app.get("/models")
def models():
    return {"available": list_models(), "active": get_active_name()}

@app.post("/models/select")
def select_model(payload: SelectModelIn):
    global model, active_name
    try:
        set_active_name(payload.name)
        active_name = payload.name
        model = load_model(active_name)
        return {"active": active_name}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo no existe. Usa GET /models")

@app.post("/predict")
def predict(payload: PenguinIn):
    try:
        X = pd.DataFrame([payload.model_dump()])
        X = pd.get_dummies(X, drop_first=True)

        # Alinear columnas con las usadas en entrenamiento
        X = X.reindex(columns=model_cols, fill_value=0)

        # Nulos numéricos
        X = X.fillna(X.median(numeric_only=True))

        pred = model.predict(X)[0]
        return {"prediction": str(pred), "model_used": active_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

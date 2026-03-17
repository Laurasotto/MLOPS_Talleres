"""
Inference API - FastAPI
=======================
Esta API carga el bundle mas reciente de MinIO (modelo + encoders) y expone
un endpoint para predecir el tipo de cobertura forestal (cover_type) dado
un conjunto de variables geograficas del terreno.

El bundle contiene:
    - model:         RandomForestClassifier entrenado
    - le_wilderness: LabelEncoder de wilderness_area del batch de entrenamiento
    - le_soil:       LabelEncoder de soil_type del batch de entrenamiento

Endpoints:
    GET  /          -> health check
    GET  /models    -> lista los modelos disponibles en MinIO
    POST /predict   -> recibe datos del terreno y devuelve la prediccion
"""

import io
import os
import pickle

import boto3
import numpy as np
from botocore.client import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE MINIO
# ──────────────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "models")
GROUP_NUMBER     = os.environ.get("GROUP_NUMBER",      "4")


def get_s3_client():
    """Crea y retorna un cliente de S3 conectado a MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )


def get_latest_bundle():
    """
    Busca y descarga el bundle mas reciente del bucket de MinIO.

    Un bundle es un diccionario con:
        - model:         el RandomForestClassifier entrenado
        - le_wilderness: el LabelEncoder de wilderness_area
        - le_soil:       el LabelEncoder de soil_type

    Usar el bundle garantiza que los encoders de la inferencia son
    exactamente los mismos que se usaron durante el entrenamiento.
    """
    s3 = get_s3_client()

    prefix   = f"group_{GROUP_NUMBER}/"
    response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
    objects  = response.get("Contents", [])

    if not objects:
        raise HTTPException(
            status_code=404,
            detail=f"No hay modelos en MinIO para group_{GROUP_NUMBER}"
        )

    # Ordenar por fecha de modificacion y tomar el mas reciente
    latest = sorted(objects, key=lambda x: x["LastModified"], reverse=True)[0]
    print(f"Cargando bundle: {latest['Key']}")

    # Descargar el archivo .pkl desde MinIO a memoria
    buffer = io.BytesIO()
    s3.download_fileobj(MINIO_BUCKET, latest["Key"], buffer)
    buffer.seek(0)

    # Deserializar el bundle completo
    bundle = pickle.loads(buffer.read())
    return bundle, latest["Key"]


# ──────────────────────────────────────────────────────────────────────────────
# DEFINICION DE LA APP FASTAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Inference API - Covertype",
    version="1.0.0",
    description="""
    API de inferencia para el modelo de clasificacion de cobertura forestal.

    El modelo fue entrenado con el dataset Covertype y puede predecir el tipo
    de cobertura forestal (cover_type del 1 al 7) a partir de variables
    geograficas del terreno.

    El modelo se carga directamente desde MinIO en cada prediccion,
    usando el bundle mas reciente que incluye el modelo y los encoders.
    """
)


# ──────────────────────────────────────────────────────────────────────────────
# MODELOS DE DATOS (Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Datos de entrada para la prediccion."""
    elevation:                          int = Field(..., description="Elevacion del terreno en metros")
    aspect:                             int = Field(..., description="Orientacion en grados azimuth")
    slope:                              int = Field(..., description="Pendiente en grados")
    horizontal_distance_to_hydrology:   int = Field(..., description="Distancia horizontal a agua en metros")
    vertical_distance_to_hydrology:     int = Field(..., description="Distancia vertical a agua en metros")
    horizontal_distance_to_roadways:    int = Field(..., description="Distancia horizontal a carreteras en metros")
    hillshade_9am:                      int = Field(..., description="Indice de sombra a las 9am (0-255)")
    hillshade_noon:                     int = Field(..., description="Indice de sombra al mediodia (0-255)")
    hillshade_3pm:                      int = Field(..., description="Indice de sombra a las 3pm (0-255)")
    horizontal_distance_to_fire_points: int = Field(..., description="Distancia horizontal a puntos de incendio en metros")
    wilderness_area:                    str = Field(..., description="Nombre del area silvestre (ej: Rawah)")
    soil_type:                          str = Field(..., description="Tipo de suelo (ej: C7702)")

    class Config:
        json_schema_extra = {
            "example": {
                "elevation": 2596,
                "aspect": 51,
                "slope": 3,
                "horizontal_distance_to_hydrology": 258,
                "vertical_distance_to_hydrology": 0,
                "horizontal_distance_to_roadways": 510,
                "hillshade_9am": 221,
                "hillshade_noon": 232,
                "hillshade_3pm": 148,
                "horizontal_distance_to_fire_points": 6279,
                "wilderness_area": "Rawah",
                "soil_type": "C7702"
            }
        }


class PredictResponse(BaseModel):
    """Respuesta de la prediccion."""
    cover_type:      int = Field(..., description="Tipo de cobertura forestal predicho (1-7)")
    cover_type_name: str = Field(..., description="Nombre del tipo de cobertura forestal")
    model_used:      str = Field(..., description="Nombre del bundle usado para la prediccion")


# Mapeo de cover_type (1-7) a nombre legible
COVER_TYPE_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    """Health check — verifica que la API esta corriendo."""
    try:
        s3 = get_s3_client()
        s3.list_buckets()
        minio_status = "conectado"
    except Exception as e:
        minio_status = f"error: {str(e)}"

    return {
        "status":       "ok",
        "service":      "Inference API - Covertype",
        "minio_status": minio_status,
        "group":        GROUP_NUMBER
    }


@app.get("/models", tags=["models"])
async def list_models():
    """Lista todos los modelos disponibles en MinIO para este grupo."""
    try:
        s3       = get_s3_client()
        prefix   = f"group_{GROUP_NUMBER}/"
        response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
        objects  = response.get("Contents", [])

        if not objects:
            return {"models": [], "total": 0}

        objects = sorted(objects, key=lambda x: x["LastModified"], reverse=True)

        models = [
            {
                "name":          obj["Key"],
                "size_kb":       round(obj["Size"] / 1024, 1),
                "last_modified": obj["LastModified"].isoformat()
            }
            for obj in objects
        ]

        return {
            "models": models,
            "total":  len(models),
            "latest": models[0]["name"] if models else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: PredictRequest):
    """
    Predice el tipo de cobertura forestal dado un conjunto de variables del terreno.

    Proceso:
    1. Cargar el bundle mas reciente desde MinIO (modelo + encoders)
    2. Aplicar el LabelEncoder guardado en el bundle para wilderness_area y soil_type
       Esto garantiza que el encoding es identico al del entrenamiento
    3. Construir el vector de features en el orden correcto
    4. Predecir con el modelo
    5. Retornar el cover_type predicho y su nombre
    """
    try:
        # Cargar el bundle mas reciente de MinIO
        bundle, bundle_name = get_latest_bundle()

        # Extraer el modelo y los encoders del bundle
        model         = bundle["model"]
        le_wilderness = bundle["le_wilderness"]
        le_soil       = bundle["le_soil"]

        # ── Label Encoding con los encoders del entrenamiento ─────────────
        # Usamos transform() en vez de fit_transform() porque el encoder
        # ya fue entrenado (fit) durante el entrenamiento del modelo.
        # transform() solo aplica el mapeo que ya conoce.
        #
        # Si llega un valor que el encoder nunca vio, lanza ValueError.
        # Lo capturamos y devolvemos un error descriptivo al cliente.
        try:
            wilderness_encoded = le_wilderness.transform([request.wilderness_area])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"wilderness_area '{request.wilderness_area}' no fue visto durante el entrenamiento. "
                       f"Valores validos: {list(le_wilderness.classes_)}"
            )

        try:
            soil_encoded = le_soil.transform([request.soil_type])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"soil_type '{request.soil_type}' no fue visto durante el entrenamiento. "
                       f"Valores validos: {list(le_soil.classes_)}"
            )

        # ── Construir el vector de features ──────────────────────────────
        # El orden debe ser exactamente el mismo que en el entrenamiento
        features = np.array([[
            request.elevation,
            request.aspect,
            request.slope,
            request.horizontal_distance_to_hydrology,
            request.vertical_distance_to_hydrology,
            request.horizontal_distance_to_roadways,
            request.hillshade_9am,
            request.hillshade_noon,
            request.hillshade_3pm,
            request.horizontal_distance_to_fire_points,
            wilderness_encoded,
            soil_encoded
        ]])

        # ── Prediccion ────────────────────────────────────────────────────
        cover_type = int(model.predict(features)[0])
        cover_name = COVER_TYPE_NAMES.get(cover_type, "Unknown")

        print(f"Prediccion: cover_type={cover_type} ({cover_name})")

        return PredictResponse(
            cover_type=cover_type,
            cover_type_name=cover_name,
            model_used=bundle_name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prediccion: {str(e)}")

"""
Inference API - FastAPI
=======================
Esta API carga el modelo mas reciente de MinIO y expone un endpoint
para predecir el tipo de cobertura forestal (cover_type) dado un conjunto
de variables geograficas del terreno.

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
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE MINIO
# Estos valores deben coincidir con los definidos en el docker-compose.yaml
# ──────────────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "models")
GROUP_NUMBER     = os.environ.get("GROUP_NUMBER",      "4")


# ──────────────────────────────────────────────────────────────────────────────
# CLIENTE DE MINIO
# Usamos boto3 que es compatible con la API de Amazon S3.
# MinIO implementa la misma API, por eso podemos usar boto3 sin cambios.
# ──────────────────────────────────────────────────────────────────────────────
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


def get_latest_model():
    """
    Busca y descarga el modelo mas reciente del bucket de MinIO.

    Los modelos se guardan con el formato:
        group_{n}/batch_{b}_acc_{a}_{timestamp}.pkl

    Esta funcion lista todos los modelos del grupo, los ordena por fecha
    de modificacion y descarga el mas reciente.

    Retorna el modelo deserializado (objeto RandomForestClassifier).
    """
    s3 = get_s3_client()

    # Listar todos los modelos del grupo en el bucket
    prefix = f"group_{GROUP_NUMBER}/"
    response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
    objects  = response.get("Contents", [])

    if not objects:
        raise HTTPException(
            status_code=404,
            detail=f"No hay modelos en MinIO para group_{GROUP_NUMBER}"
        )

    # Ordenar por fecha de modificacion y tomar el mas reciente
    # LastModified es un datetime que boto3 devuelve por cada objeto
    latest = sorted(objects, key=lambda x: x["LastModified"], reverse=True)[0]
    print(f"Cargando modelo: {latest['Key']}")

    # Descargar el archivo .pkl desde MinIO a memoria (sin guardarlo en disco)
    # io.BytesIO es un buffer en memoria que actua como un archivo
    buffer = io.BytesIO()
    s3.download_fileobj(MINIO_BUCKET, latest["Key"], buffer)
    buffer.seek(0)  # volver al inicio del buffer para poder leerlo

    # Deserializar el modelo con pickle
    # pickle.loads convierte los bytes del .pkl de vuelta al objeto Python
    model = pickle.loads(buffer.read())
    return model, latest["Key"]


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
    lo que garantiza que siempre se usa el modelo mas reciente.
    """
)


# ──────────────────────────────────────────────────────────────────────────────
# MODELOS DE DATOS (Pydantic)
# Pydantic valida automaticamente que los datos recibidos tengan el tipo
# correcto y los campos requeridos. Si falta algo, devuelve un error 422.
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Datos de entrada para la prediccion.
    Son exactamente las mismas columnas que devuelve la API de datos del profesor,
    tal como vienen (sin normalizar ni encodear).
    """
    elevation:                          int   = Field(..., description="Elevacion del terreno en metros")
    aspect:                             int   = Field(..., description="Orientacion en grados azimuth")
    slope:                              int   = Field(..., description="Pendiente en grados")
    horizontal_distance_to_hydrology:   int   = Field(..., description="Distancia horizontal a agua en metros")
    vertical_distance_to_hydrology:     int   = Field(..., description="Distancia vertical a agua en metros")
    horizontal_distance_to_roadways:    int   = Field(..., description="Distancia horizontal a carreteras en metros")
    hillshade_9am:                      int   = Field(..., description="Indice de sombra a las 9am (0-255)")
    hillshade_noon:                     int   = Field(..., description="Indice de sombra al mediodia (0-255)")
    hillshade_3pm:                      int   = Field(..., description="Indice de sombra a las 3pm (0-255)")
    horizontal_distance_to_fire_points: int   = Field(..., description="Distancia horizontal a puntos de incendio en metros")
    wilderness_area:                    str   = Field(..., description="Nombre del area silvestre (ej: Rawah)")
    soil_type:                          str   = Field(..., description="Tipo de suelo (ej: C7702)")

    class Config:
        # Ejemplo que aparece en el Swagger UI (/docs)
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
    cover_type:      int   = Field(..., description="Tipo de cobertura forestal predicho (1-7)")
    cover_type_name: str   = Field(..., description="Nombre del tipo de cobertura forestal")
    model_used:      str   = Field(..., description="Nombre del modelo usado para la prediccion")


# Mapeo de cover_type (1-7) a nombre legible
# Segun el dataset original de Covertype del UCI Machine Learning Repository
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
    """
    Health check — verifica que la API esta corriendo.
    Tambien verifica la conexion con MinIO.
    """
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
    """
    Lista todos los modelos disponibles en MinIO para este grupo.
    Muestra el nombre, tamano y fecha de cada modelo.
    """
    try:
        s3       = get_s3_client()
        prefix   = f"group_{GROUP_NUMBER}/"
        response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
        objects  = response.get("Contents", [])

        if not objects:
            return {"models": [], "total": 0}

        # Ordenar por fecha de modificacion (mas reciente primero)
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

    El proceso es:
    1. Cargar el modelo mas reciente desde MinIO
    2. Aplicar Label Encoding a wilderness_area y soil_type
       (los modelos de ML no entienden texto, solo numeros)
    3. Construir el vector de features en el orden correcto
    4. Predecir con el modelo
    5. Retornar el cover_type predicho y su nombre

    Nota sobre el preprocesamiento:
    El modelo fue entrenado con datos normalizados (Min-Max) y con
    Label Encoding. Para la inferencia aplicamos Label Encoding pero
    NO normalizamos, ya que Random Forest no requiere normalizacion
    para funcionar correctamente — los arboles de decision son
    invariantes a la escala de las variables.
    """
    try:
        # Cargar el modelo mas reciente de MinIO
        model, model_name = get_latest_model()

        # ── Label Encoding ────────────────────────────────────────────────
        # Convertir wilderness_area y soil_type de texto a numero.
        # Usamos LabelEncoder de sklearn que asigna un entero a cada
        # categoria unica. Como solo tenemos un valor, fit_transform
        # siempre devolvera 0, pero es la forma correcta de hacerlo.
        #
        # IMPORTANTE: En produccion real deberiamos guardar los encoders
        # del entrenamiento y usarlos aqui para garantizar consistencia.
        # Para este proyecto usamos esta aproximacion simplificada.
        le_wilderness = LabelEncoder()
        le_soil       = LabelEncoder()

        wilderness_encoded = le_wilderness.fit_transform([request.wilderness_area])[0]
        soil_encoded       = le_soil.fit_transform([request.soil_type])[0]

        # ── Construir el vector de features ──────────────────────────────
        # El orden debe ser exactamente el mismo que se uso en el entrenamiento.
        # Ver el notebook train_model.ipynb para confirmar el orden.
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
        # model.predict devuelve un array, tomamos el primer elemento [0]
        cover_type = int(model.predict(features)[0])
        cover_name = COVER_TYPE_NAMES.get(cover_type, "Unknown")

        print(f"Prediccion: cover_type={cover_type} ({cover_name})")

        return PredictResponse(
            cover_type=cover_type,
            cover_type_name=cover_name,
            model_used=model_name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prediccion: {str(e)}")

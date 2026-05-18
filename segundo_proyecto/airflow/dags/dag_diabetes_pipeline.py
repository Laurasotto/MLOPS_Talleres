# ─────────────────────────────────────────────────────────────────────────────
# Aquí importamos todo lo que vamos a necesitar a lo largo del DAG.
# ─────────────────────────────────────────────────────────────────────────────

# Librerías estándar de Python — manejo de archivos, JSON, hashes, fechas y logs
import os
import json
import hashlib
import logging
from datetime import datetime, timedelta

# Librerías de datos y visualización
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # usamos 'Agg' porque no hay pantalla disponible en el servidor
import matplotlib.pyplot as plt
import seaborn as sns

# SQLAlchemy nos permite conectarnos a PostgreSQL y ejecutar SQL desde Python
from sqlalchemy import create_engine, text

# Scikit-learn es la librería que usamos para dividir datos y entrenar el modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, classification_report
)

# MLflow nos permite registrar experimentos, métricas y versiones del modelo
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Airflow es el orquestador — aquí importamos lo que necesitamos para definir el DAG y sus tareas
from airflow import DAG
from airflow.operators.python import PythonOperator

# ─────────────────────────────────────────────────────────────────────────────
# Configuración general del pipeline.
# Usamos variables de entorno para no hardcodear rutas ni credenciales en el código.
# Si la variable no existe, usamos un valor por defecto que funciona dentro de Docker.
# ─────────────────────────────────────────────────────────────────────────────
DB_URI = os.getenv(
    "AIRFLOW_DB_URI",
    "postgresql://postgres:postgres123@postgres-service:5432/mlops_db"
)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
DATA_PATH = os.getenv("DATA_PATH", "/opt/airflow/data/Diabetes.csv")

# Cada ejecución del pipeline carga bloques de 15.000 filas — esto lo hacemos
# para no sobrecargar la memoria cargando el CSV completo de golpe.
BATCH_SIZE = 15_000

# Nombre del experimento y del modelo en MLflow — tienen que ser consistentes
# en todo el pipeline para que MLflow pueda agrupar los runs correctamente.
EXPERIMENT_NAME = "diabetes_readmission"
MODEL_NAME = "diabetes_readmission_model"

# El alias 'champion' lo usamos para marcar la mejor versión del modelo en el registry de MLflow.
# La API de inferencia siempre carga el modelo con este alias, no una versión específica.
CHAMPION_ALIAS = "champion"

# Lista de las 23 columnas de medicamentos en el dataset.
# Cada uno puede tener valores: "No", "Steady" (dosis estable), "Up" (aumentada), "Down" (reducida).
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Estas son las columnas que el modelo va a usar como entrada (features).
# Incluyen datos numéricos del paciente, variables codificadas de edad/raza/género,
# resultados de laboratorio, medicamentos y códigos de diagnóstico.
FEATURE_COLS = [
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

# Lo que queremos predecir: si el paciente fue readmitido en menos de 30 días (1) o no (0).
TARGET_COL = "readmitted_binary"

log = logging.getLogger(__name__)


def get_engine():
    # Creamos el motor de conexión a PostgreSQL aquí como función para que cada tarea
    # de Airflow tenga su propia conexión limpia. Esto evita problemas de conexiones
    # compartidas entre tareas que corren en procesos distintos.
    return create_engine(DB_URI)


def ensure_schemas(engine):
    # Aquí nos aseguramos de que los schemas y tablas existan en PostgreSQL
    # antes de intentar escribir en ellos. Usamos "CREATE IF NOT EXISTS"
    # para que sea seguro llamar esta función varias veces sin errores.
    # engine.begin() abre una transacción y hace commit automático al salir del bloque.
    with engine.begin() as conn:
        log.info(">>> [ensure_schemas] Creando schema 'raw_data' si no existe...")
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw_data"))
        log.info(">>> [ensure_schemas] Creando schema 'clean_data' si no existe...")
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clean_data"))
        log.info(">>> [ensure_schemas] Creando schema 'inference_logs' si no existe...")
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS inference_logs"))
        log.info(">>> [ensure_schemas] Creando tabla 'raw_data.batch_state' si no existe (tracking de batches cargados)...")
        # Esta tabla lleva el registro de qué batches ya se cargaron y hasta qué fila llegamos.
        # Así el pipeline puede pausar y continuar donde quedó sin volver a cargar todo desde el inicio.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_data.batch_state (
                id              SERIAL PRIMARY KEY,
                batch_id        INT UNIQUE NOT NULL,
                last_row_loaded INT NOT NULL DEFAULT 0,
                total_rows      INT,
                status          VARCHAR(20) DEFAULT 'in_progress',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        log.info(">>> [ensure_schemas] Creando tabla 'raw_data.diabetes_raw' si no existe (datos crudos del CSV)...")
        # Aquí guardamos los datos tal como vienen del CSV, sin modificar nada.
        # La columna row_hash es un identificador único por fila que usamos para
        # evitar cargar duplicados si el pipeline se vuelve a ejecutar.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_data.diabetes_raw (
                id                          SERIAL PRIMARY KEY,
                batch_id                    INT NOT NULL,
                load_timestamp              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_file                 VARCHAR(255),
                row_hash                    VARCHAR(64) UNIQUE,
                status                      VARCHAR(20) DEFAULT 'loaded',
                race                        TEXT,
                gender                      TEXT,
                age                         TEXT,
                weight                      TEXT,
                admission_type_id           TEXT,
                discharge_disposition_id    TEXT,
                admission_source_id         TEXT,
                time_in_hospital            TEXT,
                payer_code                  TEXT,
                medical_specialty           TEXT,
                num_lab_procedures          TEXT,
                num_procedures              TEXT,
                num_medications             TEXT,
                number_outpatient           TEXT,
                number_emergency            TEXT,
                number_inpatient            TEXT,
                diag_1                      TEXT,
                diag_2                      TEXT,
                diag_3                      TEXT,
                number_diagnoses            TEXT,
                max_glu_serum               TEXT,
                a1cresult                   TEXT,
                diabetesmed                 TEXT,
                metformin                   TEXT,
                repaglinide                 TEXT,
                nateglinide                 TEXT,
                chlorpropamide              TEXT,
                glimepiride                 TEXT,
                acetohexamide               TEXT,
                glipizide                   TEXT,
                glyburide                   TEXT,
                tolbutamide                 TEXT,
                pioglitazone                TEXT,
                rosiglitazone               TEXT,
                acarbose                    TEXT,
                miglitol                    TEXT,
                troglitazone                TEXT,
                tolazamide                  TEXT,
                examide                     TEXT,
                citoglipton                 TEXT,
                insulin                     TEXT,
                glyburide_metformin         TEXT,
                glipizide_metformin         TEXT,
                glimepiride_pioglitazone    TEXT,
                metformin_rosiglitazone     TEXT,
                metformin_pioglitazone      TEXT,
                change_col                  TEXT,
                readmitted                  TEXT,
                created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        log.info(">>> [ensure_schemas] Creando tabla 'clean_data.diabetes_clean' si no existe (datos procesados para entrenamiento)...")
        # Esta tabla guarda los datos ya procesados y codificados — los que realmente
        # usa el modelo. Todo es FLOAT porque los modelos de ML necesitan valores numéricos.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clean_data.diabetes_clean (
                id                               SERIAL PRIMARY KEY,
                batch_id                         INT NOT NULL,
                raw_id                           INT,
                time_in_hospital                 FLOAT,
                num_lab_procedures               FLOAT,
                num_procedures                   FLOAT,
                num_medications                  FLOAT,
                number_outpatient                FLOAT,
                number_emergency                 FLOAT,
                number_inpatient                 FLOAT,
                number_diagnoses                 FLOAT,
                age_encoded                      FLOAT,
                gender_encoded                   FLOAT,
                race_encoded                     FLOAT,
                admission_type_id                FLOAT,
                discharge_disposition_id         FLOAT,
                admission_source_id              FLOAT,
                a1cresult_encoded                FLOAT,
                max_glu_serum_encoded            FLOAT,
                metformin_encoded                FLOAT,
                repaglinide_encoded              FLOAT,
                nateglinide_encoded              FLOAT,
                chlorpropamide_encoded           FLOAT,
                glimepiride_encoded              FLOAT,
                acetohexamide_encoded            FLOAT,
                glipizide_encoded                FLOAT,
                glyburide_encoded                FLOAT,
                tolbutamide_encoded              FLOAT,
                pioglitazone_encoded             FLOAT,
                rosiglitazone_encoded            FLOAT,
                acarbose_encoded                 FLOAT,
                miglitol_encoded                 FLOAT,
                troglitazone_encoded             FLOAT,
                tolazamide_encoded               FLOAT,
                examide_encoded                  FLOAT,
                citoglipton_encoded              FLOAT,
                insulin_encoded                  FLOAT,
                glyburide_metformin_encoded      FLOAT,
                glipizide_metformin_encoded      FLOAT,
                glimepiride_pioglitazone_encoded FLOAT,
                metformin_rosiglitazone_encoded  FLOAT,
                metformin_pioglitazone_encoded   FLOAT,
                change_encoded                   FLOAT,
                diabetesmed_encoded              FLOAT,
                diag_1_code                      FLOAT,
                diag_2_code                      FLOAT,
                diag_3_code                      FLOAT,
                readmitted_binary                INT NOT NULL,
                created_at                       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        log.info(">>> [ensure_schemas] Todos los schemas y tablas base están listos: raw_data.batch_state, raw_data.diabetes_raw, clean_data.diabetes_clean.")


# ──────────────────────────────────────────────
# Tarea 1: Validar el archivo fuente
# Aquí verificamos que el CSV existe y tiene las columnas que esperamos
# antes de intentar cargarlo — falla rápido si algo está mal configurado.
# ──────────────────────────────────────────────
def task_validate_source(**context):
    log.info(">>> [validate_source] Verificando que el archivo fuente existe en %s", DATA_PATH)
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Mount the file into the Airflow container at this path."
        )
    log.info(">>> [validate_source] Archivo encontrado. Leyendo encabezado para verificar columnas...")
    # Leemos solo las primeras 5 filas para chequear columnas — no queremos cargar todo el CSV aquí
    df = pd.read_csv(DATA_PATH, nrows=5)
    required_cols = {"readmitted", "age", "gender", "admission_type_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Contamos las filas leyendo línea a línea para no cargar el CSV entero en memoria
    total = sum(1 for _ in open(DATA_PATH)) - 1  # restamos 1 por la fila de encabezado
    log.info(">>> [validate_source] OK — archivo validado. Columnas correctas. Total de filas estimado: %d", total)
    # Guardamos el total en XCom para que otras tareas puedan consultarlo
    context["ti"].xcom_push(key="total_rows", value=total)


# ──────────────────────────────────────────────
# Tarea 2: Cargar el siguiente batch a la tabla raw
# Aquí leemos un bloque de BATCH_SIZE filas del CSV y las insertamos en la BD.
# Usamos un hash por fila para evitar duplicados si el pipeline se vuelve a correr.
# ──────────────────────────────────────────────
def task_load_batch(**context):
    log.info(">>> [load_batch] Conectando a la base de datos PostgreSQL...")
    engine = get_engine()
    ensure_schemas(engine)
    log.info(">>> [load_batch] Schemas y tablas verificados/creados correctamente.")

    log.info(">>> [load_batch] Consultando el estado del último batch cargado...")
    with engine.connect() as conn:
        # Consultamos cuántas filas llevamos cargadas hasta ahora para saber dónde continuar
        result = conn.execute(
            text("SELECT COALESCE(MAX(batch_id), 0), COALESCE(MAX(last_row_loaded), 0) FROM raw_data.batch_state")
        ).fetchone()
        prev_batch_id, rows_loaded_so_far = result[0], result[1]
    log.info(">>> [load_batch] Último batch: %d | Filas ya cargadas: %d", prev_batch_id, rows_loaded_so_far)

    # Contamos las filas del CSV leyendo línea a línea para no cargarlo entero en RAM
    with open(DATA_PATH) as f:
        total_rows = sum(1 for _ in f) - 1  # restamos 1 por el encabezado
    log.info(">>> [load_batch] Total de filas en el CSV: %d", total_rows)

    # Si ya cargamos todas las filas, no hay nada nuevo que hacer en este batch
    if rows_loaded_so_far >= total_rows:
        log.info(">>> [load_batch] Todas las filas ya fueron cargadas. No hay datos nuevos para este batch.")
        context["ti"].xcom_push(key="batch_id", value=prev_batch_id)
        context["ti"].xcom_push(key="rows_in_batch", value=0)
        context["ti"].xcom_push(key="all_loaded", value=True)
        return

    batch_id = prev_batch_id + 1
    start_row = rows_loaded_so_far
    end_row = min(start_row + BATCH_SIZE, total_rows)
    log.info(">>> [load_batch] Preparando batch %d: filas %d a %d (%d registros a cargar)", batch_id, start_row, end_row, end_row - start_row)

    # Leemos solo las filas que aún no se han cargado.
    # skiprows salta las filas ya procesadas sin saltarse el encabezado (por eso empieza en 1).
    skip = range(1, start_row + 1) if start_row > 0 else None
    batch_df = pd.read_csv(DATA_PATH, dtype=str, skiprows=skip, nrows=BATCH_SIZE)
    log.info(">>> [load_batch] Batch leído del CSV. Calculando hashes para deduplicación...")

    log.info(
        "Loading batch %d: rows %d-%d (%d records)",
        batch_id, start_row, end_row, len(batch_df)
    )

    # Renombramos algunas columnas para que sean compatibles con los nombres en la BD.
    # Los guiones (-) y las mayúsculas (camelCase) generan problemas en SQL,
    # y 'change' es una palabra reservada en algunos dialectos SQL.
    col_rename = {
        "change": "change_col",
        "A1Cresult": "a1cresult",
        "diabetesMed": "diabetesmed",
        "glyburide-metformin": "glyburide_metformin",
        "glipizide-metformin": "glipizide_metformin",
        "glimepiride-pioglitazone": "glimepiride_pioglitazone",
        "metformin-rosiglitazone": "metformin_rosiglitazone",
        "metformin-pioglitazone": "metformin_pioglitazone",
    }
    batch_df.rename(columns=col_rename, inplace=True)

    # Calculamos un hash MD5 por cada fila concatenando todos sus valores.
    # Si el mismo registro ya está en la BD, el INSERT lo ignora gracias a
    # ON CONFLICT (row_hash) DO NOTHING — así evitamos duplicados.
    def row_hash(row):
        return hashlib.md5("|".join(str(v) for v in row).encode()).hexdigest()

    batch_df["row_hash"] = batch_df.apply(row_hash, axis=1)
    batch_df["batch_id"] = batch_id
    batch_df["load_timestamp"] = datetime.utcnow()
    batch_df["source_file"] = DATA_PATH
    batch_df["status"] = "loaded"

    # Seleccionamos solo las columnas que existen en la tabla de la BD
    db_cols = [
        "batch_id", "load_timestamp", "source_file", "row_hash", "status",
        "race", "gender", "age", "weight",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "payer_code", "medical_specialty",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "diag_1", "diag_2", "diag_3", "number_diagnoses",
        "max_glu_serum", "a1cresult", "diabetesmed",
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
        "insulin", "glyburide_metformin", "glipizide_metformin",
        "glimepiride_pioglitazone", "metformin_rosiglitazone",
        "metformin_pioglitazone", "change_col", "readmitted",
    ]

    insert_df = batch_df[[c for c in db_cols if c in batch_df.columns]]

    log.info(">>> [load_batch] Insertando %d filas en raw_data.diabetes_raw (duplicados ignorados por row_hash)...", len(insert_df))
    # Insertamos fila por fila usando ON CONFLICT DO NOTHING para manejar duplicados sin errores.
    # engine.begin() garantiza que todo el batch se confirma en una sola transacción.
    with engine.begin() as conn:
        for _, row in insert_df.iterrows():
            vals = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            cols = list(vals.keys())
            placeholders = ", ".join(f":{c}" for c in cols)
            col_str = ", ".join(cols)
            conn.execute(
                text(
                    f"INSERT INTO raw_data.diabetes_raw ({col_str}) "
                    f"VALUES ({placeholders}) ON CONFLICT (row_hash) DO NOTHING"
                ),
                vals,
            )

        # Actualizamos batch_state para registrar hasta qué fila llegamos.
        # Si el batch ya existía (por un reintento fallido), actualizamos en lugar de insertar.
        conn.execute(
            text("""
                INSERT INTO raw_data.batch_state (batch_id, last_row_loaded, total_rows, status)
                VALUES (:bid, :loaded, :total, 'completed')
                ON CONFLICT (batch_id) DO UPDATE
                    SET last_row_loaded = :loaded, total_rows = :total,
                        status = 'completed', updated_at = CURRENT_TIMESTAMP
            """),
            {"bid": batch_id, "loaded": end_row, "total": total_rows},
        )

    log.info(">>> [load_batch] Actualizando batch_state con batch_id=%d, filas_cargadas=%d/%d", batch_id, end_row, total_rows)
    log.info(">>> [load_batch] COMPLETADO — Batch %d insertado exitosamente (%d filas). Progreso total: %d/%d filas del dataset.", batch_id, len(batch_df), end_row, total_rows)

    # Compartimos el batch_id y el conteo de filas con las siguientes tareas usando XCom
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    context["ti"].xcom_push(key="rows_in_batch", value=len(batch_df))
    context["ti"].xcom_push(key="all_loaded", value=(end_row >= total_rows))


# ──────────────────────────────────────────────
# Tarea 3: Validar la calidad de los datos cargados
# Aquí revisamos que los datos del batch recién cargado tengan sentido:
# valores válidos en columnas clave y nulos esperables.
# ──────────────────────────────────────────────
def task_validate_quality(**context):
    batch_id = context["ti"].xcom_pull(key="batch_id", task_ids="load_batch")
    rows_in_batch = context["ti"].xcom_pull(key="rows_in_batch", task_ids="load_batch")

    # Si no hay filas nuevas en este batch, no hay nada que validar
    if rows_in_batch == 0:
        log.info(">>> [validate_quality] No hay filas nuevas en este batch — omitiendo validación.")
        return

    log.info(">>> [validate_quality] Conectando a la BD para leer el batch %d recién cargado...", batch_id)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM raw_data.diabetes_raw WHERE batch_id = :bid"), {"bid": batch_id})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    log.info(">>> [validate_quality] %d filas leídas de raw_data.diabetes_raw para batch %d.", len(df), batch_id)
    log.info(">>> [validate_quality] ─────────────────────────────────────────")
    log.info(">>> [validate_quality] CHEQUEO 1: Valores válidos en columna 'readmitted'")
    log.info(">>> [validate_quality]   Valores esperados: <30, >30, NO")

    # Verificamos que la columna objetivo solo tenga los tres valores posibles del dataset
    valid_readmitted = {"<30", ">30", "NO"}
    invalid_mask = ~df["readmitted"].isin(valid_readmitted)
    readmitted_counts = df["readmitted"].value_counts().to_dict()
    log.info(">>> [validate_quality]   Distribución encontrada: %s", readmitted_counts)
    if invalid_mask.sum() > 0:
        log.warning(">>> [validate_quality]   ADVERTENCIA: %d filas con valores fuera del rango esperado.", invalid_mask.sum())
    else:
        log.info(">>> [validate_quality]   Resultado: OK — todos los valores son válidos.")

    log.info(">>> [validate_quality] ─────────────────────────────────────────")
    log.info(">>> [validate_quality] CHEQUEO 2: Valores válidos en columna 'gender'")
    log.info(">>> [validate_quality]   Valores esperados: Male, Female, Unknown/Invalid")

    # Verificamos que el género solo tenga los tres valores conocidos del dataset
    valid_genders = {"Male", "Female", "Unknown/Invalid"}
    inv_gender = ~df["gender"].isin(valid_genders)
    gender_counts = df["gender"].value_counts().to_dict()
    log.info(">>> [validate_quality]   Distribución encontrada: %s", gender_counts)
    if inv_gender.sum() > 0:
        log.warning(">>> [validate_quality]   ADVERTENCIA: %d filas con valores inesperados en gender.", inv_gender.sum())
    else:
        log.info(">>> [validate_quality]   Resultado: OK — todos los valores son válidos.")

    log.info(">>> [validate_quality] ─────────────────────────────────────────")
    log.info(">>> [validate_quality] CHEQUEO 3: Valores nulos por columna")
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        for col, cnt in cols_with_nulls.items():
            log.info(">>> [validate_quality]   %s: %d nulos (%.1f%%)", col, cnt, cnt / len(df) * 100)
    else:
        log.info(">>> [validate_quality]   Sin valores nulos detectados.")
    null_pct = df.isnull().mean().max()
    log.info(">>> [validate_quality]   Porcentaje máximo de nulos en una sola columna: %.2f%%", null_pct * 100)

    log.info(">>> [validate_quality] ─────────────────────────────────────────")
    log.info(">>> [validate_quality] COMPLETADO — Batch %d validado: %d filas revisadas, %d columnas chequeadas.",
             batch_id, len(df), len(df.columns))


# ──────────────────────────────────────────────
# Tarea 4: Preprocesar y guardar los datos limpios
# Aquí transformamos los datos crudos (texto) en valores numéricos
# que el modelo de ML puede entender. Este paso se llama "feature engineering".
# ──────────────────────────────────────────────
def preprocess_df(df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    # Trabajamos sobre una copia para no modificar el DataFrame original
    df = df.copy()

    # En el dataset, los valores desconocidos están representados como '?' en lugar de NaN.
    # Los convertimos a NaN para que pandas los trate como valores faltantes.
    df.replace("?", np.nan, inplace=True)

    # Eliminamos columnas con demasiados valores faltantes o que no aportan al modelo.
    # 'weight' tiene más del 96% de nulos, 'payer_code' y 'medical_specialty' tienen
    # muchos nulos y no son buenos predictores de readmisión.
    drop_cols = ["weight", "payer_code", "medical_specialty"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Eliminamos filas con género inválido porque no podemos codificarlas de forma confiable
    df = df[df["gender"] != "Unknown/Invalid"]

    # La edad viene como rangos de texto como '[40-50)'.
    # La convertimos al punto medio del rango para tener un valor numérico continuo.
    # Por ejemplo '[40-50)' → 45. Si no reconocemos el rango, usamos 55 como valor por defecto.
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_encoded"] = df["age"].map(age_map).fillna(55).astype(float)

    # Codificamos la raza como número porque los modelos no pueden trabajar con texto.
    # Los valores que no reconocemos los agrupamos en la categoría 'Other' (4).
    race_map = {
        "Caucasian": 0, "AfricanAmerican": 1, "Hispanic": 2,
        "Asian": 3, "Other": 4,
    }
    df["race_encoded"] = df["race"].map(race_map).fillna(4).astype(float)

    # Codificamos el género como binario: Male=0, Female=1
    df["gender_encoded"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(float)

    # Convertimos las columnas numéricas a float.
    # Si hay algún valor que no se pueda convertir, lo reemplazamos con 0.
    num_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    # Codificamos los resultados de laboratorio de glucosa.
    # 'None' significa que no se midió, 'Norm' que está normal, '>7' o '>8' que está elevado.
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    df["a1cresult_encoded"] = df["a1cresult"].map(a1c_map).fillna(0).astype(float)

    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    df["max_glu_serum_encoded"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(float)

    # Codificamos cada medicamento: No=0, Steady=1 (dosis estable), Up=2 (aumentada), Down=3 (reducida).
    # Si el valor es desconocido o faltante, asumimos que no se administró (0).
    med_val_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    raw_med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin",
        "glyburide_metformin", "glipizide_metformin",
        "glimepiride_pioglitazone", "metformin_rosiglitazone",
        "metformin_pioglitazone",
    ]
    for col in raw_med_cols:
        # Buscamos la columna con guión o con guión bajo, dependiendo de cómo esté en el DataFrame
        src = col.replace("_", "-") if col.replace("_", "-") in df.columns else col
        val = df[src].map(med_val_map).fillna(0).astype(float) if src in df.columns else 0.0
        df[f"{col}_encoded"] = val

    # Codificamos si hubo cambio en medicamentos (Ch=1) y si el paciente usa medicación para diabetes (Yes=1)
    df["change_encoded"] = df["change_col"].map({"No": 0, "Ch": 1}).fillna(0).astype(float) \
        if "change_col" in df.columns else 0.0
    df["diabetesmed_encoded"] = df["diabetesmed"].map({"No": 0, "Yes": 1}).fillna(0).astype(float)

    # Los diagnósticos vienen como códigos CIE-9 (ej. "428.0", "E11", "V30").
    # Tomamos solo los primeros 3 caracteres y eliminamos letras para quedarnos con el número.
    # Si no se puede convertir (ej. "E11"), usamos 0.
    for diag_col, enc_col in [("diag_1", "diag_1_code"), ("diag_2", "diag_2_code"), ("diag_3", "diag_3_code")]:
        if diag_col in df.columns:
            df[enc_col] = pd.to_numeric(
                df[diag_col].astype(str).str[:3].str.replace("E", "").str.replace("V", ""),
                errors="coerce"
            ).fillna(0).astype(float)
        else:
            df[enc_col] = 0.0

    # Creamos la variable objetivo binaria: 1 si fue readmitido en menos de 30 días, 0 si no.
    # Esta es la que el modelo va a intentar predecir.
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    df["batch_id"] = batch_id
    return df


def task_preprocess(**context):
    batch_id = context["ti"].xcom_pull(key="batch_id", task_ids="load_batch")
    rows_in_batch = context["ti"].xcom_pull(key="rows_in_batch", task_ids="load_batch")

    # Si no hay datos nuevos en este batch, no hay nada que procesar
    if rows_in_batch == 0:
        log.info(">>> [preprocess] No hay filas nuevas — omitiendo preprocesamiento.")
        return

    log.info(">>> [preprocess] Leyendo %d filas crudas del batch %d desde raw_data.diabetes_raw...", rows_in_batch, batch_id)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM raw_data.diabetes_raw WHERE batch_id = :bid"), {"bid": batch_id})
        raw_df = pd.DataFrame(result.fetchall(), columns=result.keys())

    log.info(">>> [preprocess] Aplicando transformaciones: codificando edad, raza, género, medicamentos, diagnósticos...")
    raw_ids = raw_df["id"].tolist()
    processed = preprocess_df(raw_df, batch_id)

    # Nos quedamos solo con las columnas de features más la variable objetivo
    out_cols = ["batch_id"] + FEATURE_COLS + [TARGET_COL]
    processed = processed[[c for c in out_cols if c in processed.columns]]
    processed["raw_id"] = None

    log.info(">>> [preprocess] Transformación completada. Insertando %d filas limpias en clean_data.diabetes_clean...", len(processed))
    # Usamos INSERT fila por fila porque to_sql no es compatible con SQLAlchemy 1.4 + pandas 2.2
    records = processed.where(pd.notnull(processed), None).to_dict(orient="records")
    if records:
        cols = list(records[0].keys())
        col_str = ", ".join(cols)
        val_str = ", ".join(f":{c}" for c in cols)
        with engine.begin() as conn:
            conn.execute(text(f"INSERT INTO clean_data.diabetes_clean ({col_str}) VALUES ({val_str})"), records)

    log.info(">>> [preprocess] COMPLETADO — %d filas limpias guardadas en clean_data para batch %d.", len(processed), batch_id)
    context["ti"].xcom_push(key="clean_rows", value=len(processed))


# ──────────────────────────────────────────────
# Tarea 5: Dividir los datos en train / validación / test
# Aquí separamos los datos en tres conjuntos para entrenar y evaluar el modelo.
# Usamos todos los datos acumulados de todos los batches, no solo el batch actual.
# ──────────────────────────────────────────────
def task_split_data(**context):
    batch_id = context["ti"].xcom_pull(key="batch_id", task_ids="load_batch")

    log.info(">>> [split_data] Leyendo todos los datos limpios acumulados desde clean_data.diabetes_clean...")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM clean_data.diabetes_clean"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    log.info(">>> [split_data] %d filas totales disponibles para entrenamiento (acumulado de todos los batches).", len(df))
    log.info("Splitting %d total clean rows (split_version=%d)", len(df), batch_id)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Dividimos en 70% entrenamiento, 15% validación y 15% test.
    # Usamos stratify=y para que la proporción de casos positivos sea igual en los tres conjuntos.
    # Esto es importante porque el dataset está desbalanceado (hay muchos menos readmitidos en <30 días)
    # y sin stratify el modelo podría entrenarse o evaluarse con muestras sesgadas.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    def save_split(X_split, y_split, table_name):
        split_df = X_split.copy()
        split_df[TARGET_COL] = y_split.values
        # batch_id es NOT NULL en la tabla, lo heredamos del schema de diabetes_clean
        split_df["batch_id"] = batch_id
        split_df["split_version"] = batch_id
        # Usamos INSERT en chunks de 500 porque to_sql no es compatible con SQLAlchemy 1.4 + pandas 2.2
        records = split_df.where(pd.notnull(split_df), None).to_dict(orient="records")
        if records:
            cols = list(records[0].keys())
            col_str = ", ".join(cols)
            val_str = ", ".join(f":{c}" for c in cols)
            insert_sql = text(f"INSERT INTO clean_data.{table_name} ({col_str}) VALUES ({val_str})")
            chunk_size = 500
            with engine.begin() as conn:
                # Borramos el split anterior antes de insertar el nuevo — siempre entrenamos con el split más reciente
                conn.execute(text(f"TRUNCATE TABLE clean_data.{table_name}"))
                for i in range(0, len(records), chunk_size):
                    conn.execute(insert_sql, records[i:i + chunk_size])
        else:
            with engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE clean_data.{table_name}"))

    # Creamos las tablas de splits si no existen todavía.
    # Usamos LIKE diabetes_clean para heredar las mismas columnas y tipos de datos.
    with engine.begin() as conn:
        for tbl in ["diabetes_train", "diabetes_val", "diabetes_test"]:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS clean_data.{tbl} (
                    LIKE clean_data.diabetes_clean INCLUDING DEFAULTS,
                    split_version INT
                )
            """))

    log.info(">>> [split_data] Dividiendo: 70%% train / 15%% validación / 15%% test (stratified)...")
    log.info(">>> [split_data] Guardando splits en diabetes_train, diabetes_val, diabetes_test...")
    save_split(X_train, y_train, "diabetes_train")
    save_split(X_val, y_val, "diabetes_val")
    save_split(X_test, y_test, "diabetes_test")

    log.info(">>> [split_data] COMPLETADO — train:%d | val:%d | test:%d filas guardadas.",
        len(X_train), len(X_val), len(X_test))
    log.info(
        "Split done — train:%d val:%d test:%d",
        len(X_train), len(X_val), len(X_test)
    )
    context["ti"].xcom_push(key="train_size", value=len(X_train))
    context["ti"].xcom_push(key="val_size", value=len(X_val))
    context["ti"].xcom_push(key="test_size", value=len(X_test))


# ──────────────────────────────────────────────
# Tarea 6: Entrenar el modelo
# Aquí entrenamos un Random Forest con los datos de entrenamiento
# y calculamos las métricas sobre el conjunto de validación.
# ──────────────────────────────────────────────
def task_train_model(**context):
    log.info(">>> [train_model] Cargando datos de entrenamiento y validación desde la BD...")
    engine = get_engine()

    with engine.connect() as conn:
        r = conn.execute(text("SELECT * FROM clean_data.diabetes_train"))
        train_df = pd.DataFrame(r.fetchall(), columns=r.keys())
        r = conn.execute(text("SELECT * FROM clean_data.diabetes_val"))
        val_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    log.info(">>> [train_model] %d filas de train | %d filas de validación cargadas.", len(train_df), len(val_df))

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    # Parámetros del modelo Random Forest.
    # class_weight='balanced' es importante porque el dataset está desbalanceado
    # (hay muchos más no-readmitidos que readmitidos en <30 días).
    # Esto hace que el modelo no ignore la clase minoritaria al entrenar.
    params = {
        "n_estimators": 100,        # número de árboles en el bosque
        "max_depth": 10,            # profundidad máxima de cada árbol — limita el sobreajuste
        "class_weight": "balanced", # ajusta los pesos para compensar el desbalance de clases
        "random_state": 42,         # semilla fija para que los resultados sean reproducibles
        "n_jobs": -1,               # usa todos los núcleos del procesador disponibles
    }

    log.info(">>> [train_model] Entrenando RandomForestClassifier con %d árboles, profundidad máx %d...", params["n_estimators"], params["max_depth"])
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    log.info(">>> [train_model] Entrenamiento completado. Evaluando sobre el set de validación...")

    y_pred = model.predict(X_val)
    # predict_proba devuelve probabilidades — usamos la columna 1 (probabilidad de readmisión en <30 días)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Calculamos múltiples métricas para tener una visión completa del rendimiento
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_val, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
    }

    # Guardamos qué tan importante fue cada feature para las predicciones del modelo
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    log.info(">>> [train_model] COMPLETADO — Métricas de validación:")
    for k, v in metrics.items():
        log.info(">>>              %s = %.4f", k, v)

    # Pasamos todo a la siguiente tarea usando XCom.
    # El modelo en sí no se pasa por XCom (es demasiado grande) — se re-entrena en la tarea de registro.
    context["ti"].xcom_push(key="metrics", value=metrics)
    context["ti"].xcom_push(key="params", value=params)
    context["ti"].xcom_push(key="importances", value=importances)
    context["ti"].xcom_push(key="feature_cols", value=feature_cols)


# ──────────────────────────────────────────────
# Tarea 7: Registrar el modelo en MLflow
# Aquí re-entrenamos el modelo y lo guardamos en MLflow junto con
# sus métricas, parámetros y artefactos de evaluación (gráficos, reportes).
# ──────────────────────────────────────────────
def task_register_mlflow(**context):
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="load_batch")
    metrics = ti.xcom_pull(key="metrics", task_ids="train_model")
    params = ti.xcom_pull(key="params", task_ids="train_model")
    importances = ti.xcom_pull(key="importances", task_ids="train_model")
    feature_cols = ti.xcom_pull(key="feature_cols", task_ids="train_model")
    train_size = ti.xcom_pull(key="train_size", task_ids="split_data")
    val_size = ti.xcom_pull(key="val_size", task_ids="split_data")

    log.info(">>> [register_mlflow] Conectando al servidor MLflow en %s...", MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Aquí manejamos tres casos posibles para el experimento de MLflow:
    # 1) Existe y está activo → lo usamos directamente.
    # 2) Existe pero fue eliminado con soft delete (lifecycle='deleted') → lo restauramos.
    # 3) No existe → lo creamos.
    # Esto evita el error RESOURCE_ALREADY_EXISTS que ocurre cuando se hace reset del pipeline.
    client_tmp = MlflowClient()
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        log.info(">>> [register_mlflow] Experimento '%s' activo y listo.", EXPERIMENT_NAME)
    except Exception:
        # view_type=3 incluye experimentos activos Y eliminados con soft delete
        all_experiments = client_tmp.search_experiments(view_type=3)
        deleted_exp = next((e for e in all_experiments if e.name == EXPERIMENT_NAME), None)
        if deleted_exp:
            log.info(">>> [register_mlflow] Experimento '%s' estaba eliminado — restaurándolo...", EXPERIMENT_NAME)
            client_tmp.restore_experiment(deleted_exp.experiment_id)
            mlflow.set_experiment(EXPERIMENT_NAME)
            log.info(">>> [register_mlflow] Experimento '%s' restaurado y activo.", EXPERIMENT_NAME)
        else:
            log.info(">>> [register_mlflow] Creando nuevo experimento '%s'...", EXPERIMENT_NAME)
            mlflow.create_experiment(EXPERIMENT_NAME)
            mlflow.set_experiment(EXPERIMENT_NAME)
            log.info(">>> [register_mlflow] Experimento '%s' creado y activo.", EXPERIMENT_NAME)

    log.info(">>> [register_mlflow] Cargando datos de train/val para re-entrenar el modelo final...")
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text("SELECT * FROM clean_data.diabetes_train"))
        train_df = pd.DataFrame(r.fetchall(), columns=r.keys())
        r = conn.execute(text("SELECT * FROM clean_data.diabetes_val"))
        val_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    feat_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feat_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    # Re-entrenamos el modelo aquí para tenerlo en memoria y guardarlo en MLflow.
    # n_jobs no se registra como parámetro porque no afecta el resultado, solo la velocidad.
    rf_params = {**params}
    rf_params.pop("n_jobs", None)
    model = RandomForestClassifier(**rf_params, n_jobs=-1)
    model.fit(X_train, y_train)

    log.info(">>> [register_mlflow] Iniciando run MLflow 'batch_%d'...", batch_id)
    with mlflow.start_run(run_name=f"batch_{batch_id}") as run:
        # Registramos los parámetros del entrenamiento para tener trazabilidad completa
        mlflow.log_param("batch_id", batch_id)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
        for k, v in params.items():
            mlflow.log_param(k, v)

        log.info(">>> [register_mlflow] Registrando métricas en MLflow...")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        log.info(">>> [register_mlflow] Generando artefactos: confusion matrix, feature importances, classification report...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Guardamos la importancia de cada feature como JSON para análisis posterior
            imp_path = os.path.join(tmpdir, "feature_importances.json")
            with open(imp_path, "w") as f:
                json.dump(importances, f, indent=2)
            mlflow.log_artifact(imp_path)

            # Generamos la matriz de confusión como imagen para ver visualmente los errores del modelo
            y_pred = model.predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — Batch {batch_id}")
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            fig.savefig(cm_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(cm_path)

            # Guardamos el reporte de clasificación completo con precisión, recall y F1 por clase
            report = classification_report(y_val, y_pred, output_dict=True)
            report_path = os.path.join(tmpdir, "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)

        log.info(">>> [register_mlflow] Subiendo el modelo a MinIO (S3) a través de MLflow...")
        # Guardamos el modelo en el registry de MLflow con un ejemplo de entrada
        # para que MLflow pueda inferir automáticamente el schema de los datos de entrada.
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )

        run_id = run.info.run_id

    log.info(">>> [register_mlflow] COMPLETADO — Run '%s' registrado en MLflow para batch %d. Modelo guardado en el registry como '%s'.", run_id, batch_id, MODEL_NAME)
    ti.xcom_push(key="run_id", value=run_id)


# ──────────────────────────────────────────────
# Tarea 8: Promover el mejor modelo como champion
# Aquí comparamos el modelo recién entrenado contra el champion actual.
# Si el nuevo modelo tiene mejor F1-weighted, lo promovemos como el nuevo champion.
# La API de inferencia siempre carga el modelo con el alias 'champion'.
# ──────────────────────────────────────────────
def task_promote_champion(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="register_mlflow")
    current_metrics = ti.xcom_pull(key="metrics", task_ids="train_model")
    current_f1 = current_metrics["f1_weighted"]

    log.info(">>> [promote_champion] Conectando a MLflow para evaluar si el nuevo modelo supera al champion actual...")
    log.info(">>> [promote_champion] F1-weighted del modelo recién entrenado: %.4f", current_f1)
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Buscamos la versión del modelo que corresponde al run que acabamos de registrar
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = None
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        if v.run_id == run_id:
            latest_version = v.version
            break

    if latest_version is None:
        log.warning(">>> [promote_champion] No se encontró versión de modelo para run_id=%s", run_id)
        return

    log.info(">>> [promote_champion] Versión del modelo recién registrado: %s", latest_version)

    try:
        # Intentamos obtener el champion actual para comparar su F1-weighted
        champion_mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        champion_run = client.get_run(champion_mv.run_id)
        champion_f1 = float(champion_run.data.metrics.get("f1_weighted", 0.0))
        log.info(">>> [promote_champion] Champion actual: versión %s con F1-weighted=%.4f", champion_mv.version, champion_f1)
        log.info(">>> [promote_champion] Comparando: nuevo=%.4f vs champion=%.4f", current_f1, champion_f1)
        if current_f1 > champion_f1:
            # El nuevo modelo es mejor — lo promovemos moviendo el alias 'champion' a esta versión
            client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, latest_version)
            log.info(">>> [promote_champion] NUEVO CHAMPION — Versión %s promovida (F1: %.4f > %.4f). Alias 'champion' actualizado.", latest_version, current_f1, champion_f1)
        else:
            # El modelo actual sigue siendo mejor — dejamos el champion sin cambios
            log.info(">>> [promote_champion] El modelo nuevo (F1=%.4f) no supera al champion (F1=%.4f). Champion sin cambios.", current_f1, champion_f1)
    except Exception:
        # Si no hay champion previo (por ejemplo en la primera ejecución), promovemos este modelo directamente
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, latest_version)
        log.info(">>> [promote_champion] No existía champion previo. Versión %s promovida como primer champion (F1=%.4f).", latest_version, current_f1)


# ──────────────────────────────────────────────
# Definición del DAG
# Aquí armamos el pipeline completo: definimos los argumentos por defecto
# y encadenamos todas las tareas en el orden correcto.
# ──────────────────────────────────────────────
default_args = {
    "owner": "grupo4",
    "depends_on_past": False,       # cada ejecución es independiente de la anterior
    "retries": 1,                   # si falla, reintenta una vez automáticamente
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="dag_diabetes_pipeline",
    default_args=default_args,
    description="MLOps pipeline: ingest → preprocess → train → register champion",
    schedule_interval=None,  # se ejecuta solo cuando lo disparamos manualmente desde la UI de Airflow
    start_date=datetime(2024, 1, 1),
    catchup=False,  # no ejecuta runs pasados si el DAG estuvo apagado
    tags=["mlops", "diabetes", "grupo4"],
) as dag:

    # Tarea 1: verificar que el CSV existe y tiene las columnas correctas
    validate_source = PythonOperator(
        task_id="validate_source",
        python_callable=task_validate_source,
    )

    # Tarea 2: cargar el siguiente bloque de filas del CSV a la tabla raw
    load_batch = PythonOperator(
        task_id="load_batch",
        python_callable=task_load_batch,
    )

    # Tarea 3: revisar la calidad de los datos recién cargados
    validate_quality = PythonOperator(
        task_id="validate_quality",
        python_callable=task_validate_quality,
    )

    # Tarea 4: transformar los datos crudos en features numéricas y guardarlos limpios
    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    # Tarea 5: dividir los datos en train / validación / test (70/15/15)
    split_data = PythonOperator(
        task_id="split_data",
        python_callable=task_split_data,
    )

    # Tarea 6: entrenar el Random Forest y calcular las métricas de evaluación
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    # Tarea 7: registrar el modelo, métricas y artefactos en MLflow
    register_mlflow = PythonOperator(
        task_id="register_mlflow",
        python_callable=task_register_mlflow,
    )

    # Tarea 8: comparar con el champion actual y promover si el nuevo modelo es mejor
    promote_champion = PythonOperator(
        task_id="promote_champion",
        python_callable=task_promote_champion,
    )

    # Encadenamos las tareas en orden — cada >> indica que la tarea de la izquierda
    # debe completarse correctamente antes de que empiece la de la derecha.
    (
        validate_source
        >> load_batch
        >> validate_quality
        >> preprocess
        >> split_data
        >> train_model
        >> register_mlflow
        >> promote_champion
    )

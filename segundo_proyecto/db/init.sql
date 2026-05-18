-- ============================================================
-- MLOps Proyecto 2 - Inicialización de la base de datos
-- Este script se ejecuta una sola vez cuando postgres arranca por primera vez.
-- Crea todos los schemas y tablas que necesita el pipeline antes de que
-- Airflow o la API intenten escribir en ellos.
-- ============================================================

-- Separamos los datos en tres schemas según su etapa en el pipeline:
-- raw_data: datos tal como vienen del CSV, sin modificar
-- clean_data: datos procesados y codificados, listos para entrenar
-- inference_logs: registro de cada predicción que hace la API
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS clean_data;
CREATE SCHEMA IF NOT EXISTS inference_logs;

-- ============================================================
-- CAPA DE DATOS CRUDOS (RAW)
-- ============================================================

-- Lleva el control de qué batches ya se cargaron y hasta qué fila llegamos.
-- Esto permite que el pipeline pause y continúe sin volver a cargar desde el inicio.
CREATE TABLE IF NOT EXISTS raw_data.batch_state (
    id          SERIAL PRIMARY KEY,
    batch_id    INT UNIQUE NOT NULL,
    last_row_loaded INT NOT NULL DEFAULT 0,  -- hasta qué fila del CSV llegamos
    total_rows  INT,
    status      VARCHAR(20) DEFAULT 'in_progress',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Guarda los datos exactamente como vienen del CSV, sin transformar nada.
-- row_hash es un MD5 de cada fila — sirve para evitar insertar duplicados
-- si el pipeline se vuelve a ejecutar sobre el mismo batch.
-- Todas las columnas son TEXT porque guardamos los datos crudos sin convertir tipos.
CREATE TABLE IF NOT EXISTS raw_data.diabetes_raw (
    id                          SERIAL PRIMARY KEY,
    batch_id                    INT NOT NULL,
    load_timestamp              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_file                 VARCHAR(255),
    row_hash                    VARCHAR(64) UNIQUE,  -- MD5 para deduplicación
    status                      VARCHAR(20) DEFAULT 'loaded',
    -- columnas originales del CSV (guardadas como texto para preservar el estado crudo)
    encounter_id                TEXT,
    patient_nbr                 TEXT,
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
    diabetesmed                 TEXT,
    readmitted                  TEXT,
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para acelerar las consultas frecuentes del pipeline
CREATE INDEX IF NOT EXISTS idx_raw_batch_id ON raw_data.diabetes_raw(batch_id);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_data.diabetes_raw(status);

-- ============================================================
-- CAPA DE DATOS LIMPIOS (CLEAN)
-- ============================================================

-- Guarda los datos ya procesados y codificados — los que usa el modelo.
-- Todo es FLOAT porque los modelos de ML trabajan con números, no con texto.
-- raw_id referencia la fila original en diabetes_raw (trazabilidad).
CREATE TABLE IF NOT EXISTS clean_data.diabetes_clean (
    id                              SERIAL PRIMARY KEY,
    batch_id                        INT NOT NULL,
    raw_id                          INT,  -- referencia a la fila cruda de origen
    -- features numéricas directas del paciente
    time_in_hospital                FLOAT,
    num_lab_procedures              FLOAT,
    num_procedures                  FLOAT,
    num_medications                 FLOAT,
    number_outpatient               FLOAT,
    number_emergency                FLOAT,
    number_inpatient                FLOAT,
    number_diagnoses                FLOAT,
    -- variables categóricas codificadas como número
    age_encoded                     FLOAT,
    gender_encoded                  FLOAT,
    race_encoded                    FLOAT,
    admission_type_id               FLOAT,
    discharge_disposition_id        FLOAT,
    admission_source_id             FLOAT,
    -- resultados de laboratorio codificados
    a1cresult_encoded               FLOAT,
    max_glu_serum_encoded           FLOAT,
    -- medicamentos codificados (0=No, 1=Estable, 2=Aumentó, 3=Disminuyó)
    metformin_encoded               FLOAT,
    repaglinide_encoded             FLOAT,
    nateglinide_encoded             FLOAT,
    chlorpropamide_encoded          FLOAT,
    glimepiride_encoded             FLOAT,
    acetohexamide_encoded           FLOAT,
    glipizide_encoded               FLOAT,
    glyburide_encoded               FLOAT,
    tolbutamide_encoded             FLOAT,
    pioglitazone_encoded            FLOAT,
    rosiglitazone_encoded           FLOAT,
    acarbose_encoded                FLOAT,
    miglitol_encoded                FLOAT,
    troglitazone_encoded            FLOAT,
    tolazamide_encoded              FLOAT,
    examide_encoded                 FLOAT,
    citoglipton_encoded             FLOAT,
    insulin_encoded                 FLOAT,
    glyburide_metformin_encoded     FLOAT,
    glipizide_metformin_encoded     FLOAT,
    glimepiride_pioglitazone_encoded FLOAT,
    metformin_rosiglitazone_encoded FLOAT,
    metformin_pioglitazone_encoded  FLOAT,
    change_encoded                  FLOAT,
    diabetesmed_encoded             FLOAT,
    -- códigos de diagnóstico CIE-9 convertidos a número
    diag_1_code                     FLOAT,
    diag_2_code                     FLOAT,
    diag_3_code                     FLOAT,
    -- variable objetivo: 1 = readmitido en <30 días, 0 = no
    readmitted_binary               INT NOT NULL,
    created_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clean_batch_id ON clean_data.diabetes_clean(batch_id);

-- Tablas de splits train/val/test — heredan la misma estructura de diabetes_clean
-- más una columna split_version que indica con qué batch se generó este split.
-- El DAG las recrea (TRUNCATE + INSERT) en cada ejecución con todos los datos acumulados.
CREATE TABLE IF NOT EXISTS clean_data.diabetes_train (
    LIKE clean_data.diabetes_clean INCLUDING ALL,
    split_version INT
);

CREATE TABLE IF NOT EXISTS clean_data.diabetes_val (
    LIKE clean_data.diabetes_clean INCLUDING ALL,
    split_version INT
);

CREATE TABLE IF NOT EXISTS clean_data.diabetes_test (
    LIKE clean_data.diabetes_clean INCLUDING ALL,
    split_version INT
);

-- ============================================================
-- LOGS DE INFERENCIA
-- ============================================================

-- pgcrypto nos da gen_random_uuid() para generar UUIDs únicos por request
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Guarda cada predicción que hace la API — útil para auditar el modelo
-- y detectar drift (cuando las predicciones cambian con el tiempo).
CREATE TABLE IF NOT EXISTS inference_logs.predictions (
    id                  SERIAL PRIMARY KEY,
    request_id          UUID DEFAULT gen_random_uuid(),  -- ID único por request para trazabilidad
    inference_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_data          JSONB,          -- los datos del paciente que se enviaron a la API
    prediction          INT,            -- 0 o 1
    prediction_label    VARCHAR(50),    -- "readmitted_early" o "not_readmitted_early"
    probability         FLOAT,          -- confianza del modelo en la predicción
    model_name          VARCHAR(255),
    model_version       VARCHAR(50),
    model_alias         VARCHAR(50),
    response_time_ms    FLOAT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para consultas frecuentes sobre los logs (por fecha y por modelo)
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON inference_logs.predictions(inference_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON inference_logs.predictions(model_name, model_version);

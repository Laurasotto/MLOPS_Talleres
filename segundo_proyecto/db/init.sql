-- ============================================================
-- MLOps Proyecto 2 - Database Initialization
-- ============================================================

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS clean_data;
CREATE SCHEMA IF NOT EXISTS inference_logs;

-- ============================================================
-- RAW DATA LAYER
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_data.batch_state (
    id          SERIAL PRIMARY KEY,
    batch_id    INT UNIQUE NOT NULL,
    last_row_loaded INT NOT NULL DEFAULT 0,
    total_rows  INT,
    status      VARCHAR(20) DEFAULT 'in_progress',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_data.diabetes_raw (
    id                          SERIAL PRIMARY KEY,
    batch_id                    INT NOT NULL,
    load_timestamp              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_file                 VARCHAR(255),
    row_hash                    VARCHAR(64) UNIQUE,
    status                      VARCHAR(20) DEFAULT 'loaded',
    -- original CSV columns (stored as text to preserve raw state)
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

CREATE INDEX IF NOT EXISTS idx_raw_batch_id ON raw_data.diabetes_raw(batch_id);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_data.diabetes_raw(status);

-- ============================================================
-- CLEAN DATA LAYER
-- ============================================================

CREATE TABLE IF NOT EXISTS clean_data.diabetes_clean (
    id                              SERIAL PRIMARY KEY,
    batch_id                        INT NOT NULL,
    raw_id                          INT,
    -- numerical features (after preprocessing)
    time_in_hospital                FLOAT,
    num_lab_procedures              FLOAT,
    num_procedures                  FLOAT,
    num_medications                 FLOAT,
    number_outpatient               FLOAT,
    number_emergency                FLOAT,
    number_inpatient                FLOAT,
    number_diagnoses                FLOAT,
    -- encoded categorical features
    age_encoded                     FLOAT,
    gender_encoded                  FLOAT,
    race_encoded                    FLOAT,
    admission_type_id               FLOAT,
    discharge_disposition_id        FLOAT,
    admission_source_id             FLOAT,
    -- lab results
    a1cresult_encoded               FLOAT,
    max_glu_serum_encoded           FLOAT,
    -- medication features
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
    -- diagnosis codes (numeric)
    diag_1_code                     FLOAT,
    diag_2_code                     FLOAT,
    diag_3_code                     FLOAT,
    -- target variable
    readmitted_binary               INT NOT NULL,
    created_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clean_batch_id ON clean_data.diabetes_clean(batch_id);

-- Train/Val/Test split tables
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
-- INFERENCE LOGS
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS inference_logs.predictions (
    id                  SERIAL PRIMARY KEY,
    request_id          UUID DEFAULT gen_random_uuid(),
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
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON inference_logs.predictions(inference_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON inference_logs.predictions(model_name, model_version);

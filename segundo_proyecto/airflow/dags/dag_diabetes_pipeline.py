import os
import json
import hashlib
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from airflow import DAG
from airflow.operators.python import PythonOperator

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DB_URI = os.getenv(
    "AIRFLOW_DB_URI",
    "postgresql://postgres:postgres123@postgres-service:5432/mlops_db"
)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
DATA_PATH = os.getenv("DATA_PATH", "/opt/airflow/data/Diabetes.csv")
BATCH_SIZE = 15_000
EXPERIMENT_NAME = "diabetes_readmission"
MODEL_NAME = "diabetes_readmission_model"
CHAMPION_ALIAS = "champion"

# Medication columns in the dataset
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

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
TARGET_COL = "readmitted_binary"

log = logging.getLogger(__name__)


def get_engine():
    return create_engine(DB_URI)


def ensure_schemas(engine):
    # engine.begin() auto-commits on exit (SQLAlchemy 1.4 — conn.commit() not available without future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw_data"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clean_data"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS inference_logs"))
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


# ──────────────────────────────────────────────
# Task 1: Validate source file
# ──────────────────────────────────────────────
def task_validate_source(**context):
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Mount the file into the Airflow container at this path."
        )
    df = pd.read_csv(DATA_PATH, nrows=5)
    required_cols = {"readmitted", "age", "gender", "admission_type_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    total = sum(1 for _ in open(DATA_PATH)) - 1  # subtract header
    log.info("Source file validated. Estimated rows: %d", total)
    context["ti"].xcom_push(key="total_rows", value=total)


# ──────────────────────────────────────────────
# Task 2: Load batch into raw table
# ──────────────────────────────────────────────
def task_load_batch(**context):
    engine = get_engine()
    ensure_schemas(engine)

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COALESCE(MAX(batch_id), 0), COALESCE(MAX(last_row_loaded), 0) FROM raw_data.batch_state")
        ).fetchone()
        prev_batch_id, rows_loaded_so_far = result[0], result[1]

    # Count rows by streaming the file — avoids loading the full CSV into RAM
    with open(DATA_PATH) as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    if rows_loaded_so_far >= total_rows:
        log.info("All %d rows already loaded. Skipping batch load.", total_rows)
        context["ti"].xcom_push(key="batch_id", value=prev_batch_id)
        context["ti"].xcom_push(key="rows_in_batch", value=0)
        context["ti"].xcom_push(key="all_loaded", value=True)
        return

    batch_id = prev_batch_id + 1
    start_row = rows_loaded_so_far
    end_row = min(start_row + BATCH_SIZE, total_rows)

    # Read only the batch rows — skiprows skips already-loaded data rows (keeps header)
    skip = range(1, start_row + 1) if start_row > 0 else None
    batch_df = pd.read_csv(DATA_PATH, dtype=str, skiprows=skip, nrows=BATCH_SIZE)

    log.info(
        "Loading batch %d: rows %d-%d (%d records)",
        batch_id, start_row, end_row, len(batch_df)
    )

    # Rename columns to match DB schema (handle dashes, camelCase, reserved words)
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

    # Compute row hash for deduplication
    def row_hash(row):
        return hashlib.md5("|".join(str(v) for v in row).encode()).hexdigest()

    batch_df["row_hash"] = batch_df.apply(row_hash, axis=1)
    batch_df["batch_id"] = batch_id
    batch_df["load_timestamp"] = datetime.utcnow()
    batch_df["source_file"] = DATA_PATH
    batch_df["status"] = "loaded"

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

    # engine.begin() auto-commits on exit (SQLAlchemy 1.4 compatibility)
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

    log.info("Batch %d loaded successfully (%d rows).", batch_id, len(batch_df))
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    context["ti"].xcom_push(key="rows_in_batch", value=len(batch_df))
    context["ti"].xcom_push(key="all_loaded", value=(end_row >= total_rows))


# ──────────────────────────────────────────────
# Task 3: Validate data quality
# ──────────────────────────────────────────────
def task_validate_quality(**context):
    batch_id = context["ti"].xcom_pull(key="batch_id", task_ids="load_batch")
    rows_in_batch = context["ti"].xcom_pull(key="rows_in_batch", task_ids="load_batch")

    if rows_in_batch == 0:
        log.info("No new rows in batch — skipping quality validation.")
        return

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM raw_data.diabetes_raw WHERE batch_id = :bid"), {"bid": batch_id})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    log.info("Quality check on %d rows for batch %d", len(df), batch_id)

    # Check readmitted values
    valid_readmitted = {"<30", ">30", "NO"}
    invalid_mask = ~df["readmitted"].isin(valid_readmitted)
    if invalid_mask.sum() > 0:
        log.warning("%d rows with unexpected readmitted values", invalid_mask.sum())

    # Check gender
    valid_genders = {"Male", "Female", "Unknown/Invalid"}
    inv_gender = ~df["gender"].isin(valid_genders)
    if inv_gender.sum() > 0:
        log.warning("%d rows with unexpected gender values", inv_gender.sum())

    null_pct = df.isnull().mean().max()
    log.info("Max null percentage across columns: %.2f%%", null_pct * 100)

    log.info("Quality validation complete for batch %d", batch_id)


# ──────────────────────────────────────────────
# Task 4: Preprocess and store clean data
# ──────────────────────────────────────────────
def preprocess_df(df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    df = df.copy()

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop high-missing / irrelevant columns
    drop_cols = ["weight", "payer_code", "medical_specialty"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Remove invalid gender
    df = df[df["gender"] != "Unknown/Invalid"]

    # Age: '[0-10)' → midpoint int
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_encoded"] = df["age"].map(age_map).fillna(55).astype(float)

    # Race
    race_map = {
        "Caucasian": 0, "AfricanAmerican": 1, "Hispanic": 2,
        "Asian": 3, "Other": 4,
    }
    df["race_encoded"] = df["race"].map(race_map).fillna(4).astype(float)

    # Gender
    df["gender_encoded"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(float)

    # Numeric columns
    num_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    # Lab results
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    df["a1cresult_encoded"] = df["a1cresult"].map(a1c_map).fillna(0).astype(float)

    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    df["max_glu_serum_encoded"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(float)

    # Medications
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
        src = col.replace("_", "-") if col.replace("_", "-") in df.columns else col
        val = df[src].map(med_val_map).fillna(0).astype(float) if src in df.columns else 0.0
        df[f"{col}_encoded"] = val

    # Change and diabetesmed
    df["change_encoded"] = df["change_col"].map({"No": 0, "Ch": 1}).fillna(0).astype(float) \
        if "change_col" in df.columns else 0.0
    df["diabetesmed_encoded"] = df["diabetesmed"].map({"No": 0, "Yes": 1}).fillna(0).astype(float)

    # Diagnosis codes: take first 3 chars, convert to float
    for diag_col, enc_col in [("diag_1", "diag_1_code"), ("diag_2", "diag_2_code"), ("diag_3", "diag_3_code")]:
        if diag_col in df.columns:
            df[enc_col] = pd.to_numeric(
                df[diag_col].astype(str).str[:3].str.replace("E", "").str.replace("V", ""),
                errors="coerce"
            ).fillna(0).astype(float)
        else:
            df[enc_col] = 0.0

    # Target: binary readmission (<30 days = 1, else = 0)
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    df["batch_id"] = batch_id
    return df


def task_preprocess(**context):
    batch_id = context["ti"].xcom_pull(key="batch_id", task_ids="load_batch")
    rows_in_batch = context["ti"].xcom_pull(key="rows_in_batch", task_ids="load_batch")

    if rows_in_batch == 0:
        log.info("No new rows — skipping preprocessing.")
        return

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM raw_data.diabetes_raw WHERE batch_id = :bid"), {"bid": batch_id})
        raw_df = pd.DataFrame(result.fetchall(), columns=result.keys())

    log.info("Preprocessing %d raw rows for batch %d", len(raw_df), batch_id)

    # Preserve raw_id mapping
    raw_ids = raw_df["id"].tolist()
    processed = preprocess_df(raw_df, batch_id)

    # Select only the feature columns + target + metadata
    out_cols = ["batch_id"] + FEATURE_COLS + [TARGET_COL]
    processed = processed[[c for c in out_cols if c in processed.columns]]
    processed["raw_id"] = None  # we don't do row-level mapping here

    # to_sql incompatible with SQLAlchemy 1.4 + pandas 2.2 — use bulk INSERT via conn.execute
    records = processed.where(pd.notnull(processed), None).to_dict(orient="records")
    if records:
        cols = list(records[0].keys())
        col_str = ", ".join(cols)
        val_str = ", ".join(f":{c}" for c in cols)
        with engine.begin() as conn:
            conn.execute(text(f"INSERT INTO clean_data.diabetes_clean ({col_str}) VALUES ({val_str})"), records)

    log.info("Stored %d clean rows for batch %d", len(processed), batch_id)
    context["ti"].xcom_push(key="clean_rows", value=len(processed))


# ──────────────────────────────────────────────
# Task 5: Split data into train/val/test
# ──────────────────────────────────────────────
def task_split_data(**context):
    batch_id = context["ti"].xcom_pull(key="batch_id", task_ids="load_batch")

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM clean_data.diabetes_clean"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    log.info("Splitting %d total clean rows (split_version=%d)", len(df), batch_id)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    def save_split(X_split, y_split, table_name):
        split_df = X_split.copy()
        split_df[TARGET_COL] = y_split.values
        split_df["batch_id"] = batch_id  # NOT NULL constraint inherited from diabetes_clean via LIKE
        split_df["split_version"] = batch_id
        # to_sql incompatible with SQLAlchemy 1.4 + pandas 2.2 — chunked INSERT via conn.execute
        records = split_df.where(pd.notnull(split_df), None).to_dict(orient="records")
        if records:
            cols = list(records[0].keys())
            col_str = ", ".join(cols)
            val_str = ", ".join(f":{c}" for c in cols)
            insert_sql = text(f"INSERT INTO clean_data.{table_name} ({col_str}) VALUES ({val_str})")
            chunk_size = 500
            with engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE clean_data.{table_name}"))
                for i in range(0, len(records), chunk_size):
                    conn.execute(insert_sql, records[i:i + chunk_size])
        else:
            with engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE clean_data.{table_name}"))

    # Ensure split tables exist; engine.begin() auto-commits on exit (SQLAlchemy 1.4 compatibility)
    with engine.begin() as conn:
        for tbl in ["diabetes_train", "diabetes_val", "diabetes_test"]:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS clean_data.{tbl} (
                    LIKE clean_data.diabetes_clean INCLUDING DEFAULTS,
                    split_version INT
                )
            """))

    save_split(X_train, y_train, "diabetes_train")
    save_split(X_val, y_val, "diabetes_val")
    save_split(X_test, y_test, "diabetes_test")

    log.info(
        "Split done — train:%d val:%d test:%d",
        len(X_train), len(X_val), len(X_test)
    )
    context["ti"].xcom_push(key="train_size", value=len(X_train))
    context["ti"].xcom_push(key="val_size", value=len(X_val))
    context["ti"].xcom_push(key="test_size", value=len(X_test))


# ──────────────────────────────────────────────
# Task 6: Train model
# ──────────────────────────────────────────────
def task_train_model(**context):
    engine = get_engine()

    with engine.connect() as conn:
        r = conn.execute(text("SELECT * FROM clean_data.diabetes_train"))
        train_df = pd.DataFrame(r.fetchall(), columns=r.keys())
        r = conn.execute(text("SELECT * FROM clean_data.diabetes_val"))
        val_df = pd.DataFrame(r.fetchall(), columns=r.keys())

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[TARGET_COL]

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    log.info("Training RandomForestClassifier on %d samples...", len(X_train))
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_val, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
    }

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    log.info("Validation metrics: %s", metrics)
    context["ti"].xcom_push(key="metrics", value=metrics)
    context["ti"].xcom_push(key="params", value=params)
    context["ti"].xcom_push(key="importances", value=importances)
    context["ti"].xcom_push(key="feature_cols", value=feature_cols)


# ──────────────────────────────────────────────
# Task 7: Register in MLflow
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

    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception:
        mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

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

    rf_params = {**params}
    rf_params.pop("n_jobs", None)
    model = RandomForestClassifier(**rf_params, n_jobs=-1)
    model.fit(X_train, y_train)

    with mlflow.start_run(run_name=f"batch_{batch_id}") as run:
        # Log params
        mlflow.log_param("batch_id", batch_id)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Feature importance artifact
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            imp_path = os.path.join(tmpdir, "feature_importances.json")
            with open(imp_path, "w") as f:
                json.dump(importances, f, indent=2)
            mlflow.log_artifact(imp_path)

            # Confusion matrix plot
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

            # Classification report
            report = classification_report(y_val, y_pred, output_dict=True)
            report_path = os.path.join(tmpdir, "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )

        run_id = run.info.run_id

    log.info("MLflow run %s registered for batch %d", run_id, batch_id)
    ti.xcom_push(key="run_id", value=run_id)


# ──────────────────────────────────────────────
# Task 8: Promote champion model
# ──────────────────────────────────────────────
def task_promote_champion(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="register_mlflow")
    current_metrics = ti.xcom_pull(key="metrics", task_ids="train_model")
    current_f1 = current_metrics["f1_weighted"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Get the latest version of the registered model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = None
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        if v.run_id == run_id:
            latest_version = v.version
            break

    if latest_version is None:
        log.warning("Could not find model version for run_id=%s", run_id)
        return

    # Check if champion exists
    try:
        champion_mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        champion_run = client.get_run(champion_mv.run_id)
        champion_f1 = float(champion_run.data.metrics.get("f1_weighted", 0.0))
        log.info(
            "Champion f1_weighted=%.4f | Current f1_weighted=%.4f",
            champion_f1, current_f1
        )
        if current_f1 > champion_f1:
            client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, latest_version)
            log.info(
                "New champion: version %s (f1=%.4f > %.4f)",
                latest_version, current_f1, champion_f1
            )
        else:
            log.info(
                "Current model (f1=%.4f) did not beat champion (f1=%.4f). Champion unchanged.",
                current_f1, champion_f1
            )
    except Exception:
        # No champion yet — promote current model
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, latest_version)
        log.info(
            "No existing champion. Promoting version %s as champion (f1=%.4f).",
            latest_version, current_f1
        )


# ──────────────────────────────────────────────
# DAG definition
# ──────────────────────────────────────────────
default_args = {
    "owner": "grupo4",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="dag_diabetes_pipeline",
    default_args=default_args,
    description="MLOps pipeline: ingest → preprocess → train → register champion",
    schedule_interval=None,  # manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "diabetes", "grupo4"],
) as dag:

    validate_source = PythonOperator(
        task_id="validate_source",
        python_callable=task_validate_source,
    )

    load_batch = PythonOperator(
        task_id="load_batch",
        python_callable=task_load_batch,
    )

    validate_quality = PythonOperator(
        task_id="validate_quality",
        python_callable=task_validate_quality,
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    split_data = PythonOperator(
        task_id="split_data",
        python_callable=task_split_data,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    register_mlflow = PythonOperator(
        task_id="register_mlflow",
        python_callable=task_register_mlflow,
    )

    promote_champion = PythonOperator(
        task_id="promote_champion",
        python_callable=task_promote_champion,
    )

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

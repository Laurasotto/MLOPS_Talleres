from __future__ import annotations

import os
from datetime import datetime
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

import mysql.connector
import pandas as pd
import seaborn as sns


# =========================
# CONFIG (MySQL)
# =========================
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql_data")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB", "data_db")
MYSQL_USER = os.getenv("MYSQL_USER", "data_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "data_pass")

RAW_TABLE = "penguins_raw"
POST_TABLE = "post_procesados"


# =========================
# HELPERS
# =========================
def mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        autocommit=True,
    )


def pretty_banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def table_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        """,
        (MYSQL_DB, table_name),
    )
    return cur.fetchone()[0] > 0


def row_count(cur, table_name: str):
    if not table_exists(cur, table_name):
        return None
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    return int(cur.fetchone()[0])


def show_counts(cur, table_name: str, label: str):
    cnt = row_count(cur, table_name)
    if cnt is None:
        print(f"[{label}] {table_name}: (no existe)")
    else:
        print(f"[{label}] {table_name}: {cnt} rows")


# =========================
# TASKS
# =========================
def wipe_db():
    pretty_banner("TAREA 1: wipe_db (reset completo)")

    conn = mysql_conn()
    cur = conn.cursor()

    show_counts(cur, RAW_TABLE, "ANTES RAW")
    show_counts(cur, POST_TABLE, "ANTES POST")

    # RAW → solo truncar
    if table_exists(cur, RAW_TABLE):
        print(f"TRUNCATE {RAW_TABLE}")
        cur.execute(f"TRUNCATE TABLE {RAW_TABLE}")

    # POST_PROCESADOS → drop + recreate
    print(f"Recreando tabla {POST_TABLE}...")

    cur.execute(f"DROP TABLE IF EXISTS {POST_TABLE}")

    # Se crea vacía (estructura mínima)
    cur.execute(
        f"""
        CREATE TABLE {POST_TABLE} (
            dummy INT NULL
        )
        """
    )

    print(f"Tabla {POST_TABLE} recreada vacía.")

    show_counts(cur, RAW_TABLE, "DESPUES RAW")
    show_counts(cur, POST_TABLE, "DESPUES POST")

    cur.close()
    conn.close()


def load_penguins():
    pretty_banner("TAREA 2: load_penguins RAW")

    df = sns.load_dataset("penguins")

    conn = mysql_conn()
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
            species VARCHAR(32),
            island VARCHAR(32),
            bill_length_mm DOUBLE NULL,
            bill_depth_mm DOUBLE NULL,
            flipper_length_mm DOUBLE NULL,
            body_mass_g DOUBLE NULL,
            sex VARCHAR(16)
        )
        """
    )

    df2 = df.where(pd.notnull(df), None)

    rows = df2[
        [
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    ].values.tolist()

    cur.executemany(
        f"""
        INSERT INTO {RAW_TABLE}
        (species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        rows,
    )

    show_counts(cur, RAW_TABLE, "DESPUES CARGA RAW")

    cur.close()
    conn.close()


def run_model_script():
    pretty_banner("TAREA 3: Ejecutando modelo1.py")

    conn = mysql_conn()
    cur = conn.cursor()
    show_counts(cur, RAW_TABLE, "ANTES RAW")
    show_counts(cur, POST_TABLE, "ANTES POST")
    cur.close()
    conn.close()

    try:
        res = subprocess.run(
            ["python", "/opt/airflow/scripts/modelo1.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("STDOUT modelo1.py:\n", res.stdout)
        print("STDERR modelo1.py:\n", res.stderr)

    except subprocess.CalledProcessError as e:
        print(" FALLÓ modelo1.py")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    conn = mysql_conn()
    cur = conn.cursor()
    show_counts(cur, POST_TABLE, "DESPUES POST")
    cur.close()
    conn.close()

# =========================
# DAG
# =========================
with DAG(
    dag_id="dag_pipeline_penguins_raw_postprocesado",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["taller", "penguins"],
) as dag:

    t1 = PythonOperator(task_id="wipe_db", python_callable=wipe_db)
    t2 = PythonOperator(task_id="load_penguins_raw", python_callable=load_penguins)
    t3 = PythonOperator(task_id="run_modelo1", python_callable=run_model_script)

    t1 >> t2 >> t3

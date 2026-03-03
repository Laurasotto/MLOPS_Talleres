from __future__ import annotations

import os
from datetime import datetime

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

RAW_TABLE = os.getenv("RAW_TABLE", "penguins_raw")


# =========================
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


def row_count(cur, table_name: str) -> int | None:
    if not table_exists(cur, table_name):
        return None
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    return int(cur.fetchone()[0])


def show_counts(cur, label: str):
    cnt = row_count(cur, RAW_TABLE)
    if cnt is None:
        print(f"[{label}] {RAW_TABLE}: (no existe)")
    else:
        print(f"[{label}] {RAW_TABLE}: {cnt} rows")


def show_context(cur):
    print("-" * 70)
    print("Contexto de conexión MySQL")
    print("-" * 70)

    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]

    cur.execute("SELECT DATABASE()")
    current_db = cur.fetchone()[0]

    cur.execute("SELECT CURRENT_USER()")
    current_user = cur.fetchone()[0]

    cur.execute("SELECT @@hostname")
    hostname = cur.fetchone()[0]

    print(f"Servidor MySQL     : {hostname}")
    print(f"Versión MySQL      : {version}")
    print(f"Usuario actual     : {current_user}")
    print(f"Base de datos      : {current_db}")
    print(f"Host conexión      : {MYSQL_HOST}:{MYSQL_PORT}")

    cur.execute(
        """
        SELECT 
            COUNT(*) AS total_tables,
            ROUND(SUM(data_length + index_length)/1024/1024,2) AS total_mb
        FROM information_schema.tables
        WHERE table_schema = %s
        """,
        (MYSQL_DB,),
    )
    total_tables, total_mb = cur.fetchone()
    total_mb = total_mb or 0

    print(f"Total tablas       : {total_tables}")
    print(f"Tamaño schema (MB) : {total_mb}")
    print("-" * 70)


# =========================
# TASKS
# =========================
def wipe_db():
    pretty_banner("TAREA 1: wipe_db (borrar contenido BD)")

    conn = mysql_conn()
    cur = conn.cursor()

    show_context(cur)
    show_counts(cur, "ANTES")

    if table_exists(cur, RAW_TABLE):
        print(f"Borrando contenido de {RAW_TABLE} ...")
        cur.execute(f"TRUNCATE TABLE {RAW_TABLE}")
        print("Listo: TRUNCATE ejecutado.")
    else:
        print(f"No borro nada porque la tabla {RAW_TABLE} no existe aún.")

    show_counts(cur, "DESPUES")

    cur.close()
    conn.close()


def load_penguins():
    pretty_banner("TAREA 2: load_penguins (cargar penguins RAW, sin preprocesamiento)")

    df = sns.load_dataset("penguins")  # viene con NaNs, así lo dejamos
    print(f"Dataset penguins cargado en memoria: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("Primeras 3 filas (para verificar):")
    print(df.head(3))

    conn = mysql_conn()
    cur = conn.cursor()

    show_context(cur)
    show_counts(cur, "ANTES DE CARGAR")

    # crear tabla si no existe (tipos simples)
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

    # Convertimos NaN a None para mysql-connector
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

    print(f"Insertando {len(rows)} filas en {RAW_TABLE} ...")

    cur.executemany(
        f"""
        INSERT INTO {RAW_TABLE}
        (species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        rows,
    )

    print("Inserción terminada.")
    show_counts(cur, "DESPUES DE CARGAR")

    cur.close()
    conn.close()


# =========================
# DAG
# =========================
with DAG(
    dag_id="dag_pipeline_penguins_raw",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["taller", "penguins"],
) as dag:

    t1 = PythonOperator(task_id="wipe_db", python_callable=wipe_db)
    t2 = PythonOperator(task_id="load_penguins_raw", python_callable=load_penguins)

    t1 >> t2

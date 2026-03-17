from datetime import datetime
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


API_BASE_URL = "http://host.docker.internal:8080"
GROUP_NUMBER = 4

# Nombres de las tablas del pipeline (deben coincidir con el DAG principal)
TABLE_NAME      = "training_data"
PROCESSED_TABLE = "training_data_processed"
READY_TABLE     = "training_data_ready"
POSTGRES_CONN_ID = "postgres_default"


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 1: restart_api
# Reinicia el contador de batches de la API externa del profesor.
# Despues de esto, la API volvera a servir desde el batch 1.
# ══════════════════════════════════════════════════════════════════════════════
def restart_api():
    print(f"Reiniciando generacion de datos para group_number={GROUP_NUMBER}...")

    response = requests.get(
        f"{API_BASE_URL}/restart_data_generation",
        params={"group_number": GROUP_NUMBER},
        timeout=30
    )
    response.raise_for_status()

    print(f"Respuesta de la API: {response.json()}")
    print("Contador de batches reiniciado. La API volvera a servir desde batch 1.")


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 2: drop_tables
# Borra las tres tablas de PostgreSQL para empezar desde cero.
# Util cuando quieres reiniciar el pipeline completo sin datos anteriores.
#
# El orden del DROP importa: hay que borrar primero las tablas que dependen
# de training_data antes de borrar training_data misma.
# ══════════════════════════════════════════════════════════════════════════════
def drop_tables():
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    print("Borrando tablas de PostgreSQL...")
    hook.run(f"DROP TABLE IF EXISTS public.{READY_TABLE};")
    hook.run(f"DROP TABLE IF EXISTS public.{PROCESSED_TABLE};")
    hook.run(f"DROP TABLE IF EXISTS public.{TABLE_NAME};")
    print(f"Tablas eliminadas:")
    print(f"  - {READY_TABLE}")
    print(f"  - {PROCESSED_TABLE}")
    print(f"  - {TABLE_NAME}")
    print("Listo. Ya puedes activar el DAG principal para una nueva ronda de 10 batches.")


with DAG(
    dag_id="dag_restart_data",
    start_date=datetime(2026, 3, 11),
    schedule=None,   # solo ejecucion manual
    catchup=False,
    tags=["mlops", "utils"]
) as dag:

    task_restart_api = PythonOperator(
        task_id="restart_api",
        python_callable=restart_api
    )

    task_drop_tables = PythonOperator(
        task_id="drop_tables",
        python_callable=drop_tables
    )

    # Primero reinicia la API, luego borra las tablas
    task_restart_api >> task_drop_tables

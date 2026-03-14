from datetime import datetime
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator


API_BASE_URL = "http://host.docker.internal:8080"
GROUP_NUMBER = 4


def restart_data():
    print(f"Reiniciando generación de datos para group_number={GROUP_NUMBER}...")

    response = requests.get(
        f"{API_BASE_URL}/restart_data_generation",
        params={"group_number": GROUP_NUMBER},
        timeout=30
    )
    response.raise_for_status()

    print(f"Respuesta de la API: {response.json()}")
    print("Reinicio completado. Ya puedes correr el DAG principal de nuevo.")


with DAG(
    dag_id="dag_restart_data",
    start_date=datetime(2026, 3, 11),
    schedule=None,
    catchup=False,
    tags=["mlops", "utils"]
) as dag:

    task_restart = PythonOperator(
        task_id="restart_data_generation",
        python_callable=restart_data
    )

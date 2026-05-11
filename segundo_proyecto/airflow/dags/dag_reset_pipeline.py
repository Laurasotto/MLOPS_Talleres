import os
import logging
from datetime import datetime

from sqlalchemy import create_engine, text
from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow
from mlflow.tracking import MlflowClient

DB_URI = os.getenv("AIRFLOW_DB_URI", "postgresql://postgres:postgres123@postgres-service:5432/mlops_db")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = "diabetes_readmission_model"
EXPERIMENT_NAME = "diabetes_readmission"

log = logging.getLogger(__name__)


def task_reset_database(**context):
    log.info(">>> [reset_database] Conectando a PostgreSQL en mlops_db...")
    engine = create_engine(DB_URI)

    with engine.begin() as conn:
        log.info(">>> [reset_database] Truncando clean_data.diabetes_train...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_train CASCADE"))

        log.info(">>> [reset_database] Truncando clean_data.diabetes_val...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_val CASCADE"))

        log.info(">>> [reset_database] Truncando clean_data.diabetes_test...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_test CASCADE"))

        log.info(">>> [reset_database] Truncando clean_data.diabetes_clean...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_clean CASCADE"))

        log.info(">>> [reset_database] Truncando raw_data.diabetes_raw...")
        conn.execute(text("TRUNCATE TABLE raw_data.diabetes_raw CASCADE"))

        log.info(">>> [reset_database] Truncando raw_data.batch_state...")
        conn.execute(text("TRUNCATE TABLE raw_data.batch_state CASCADE"))

    log.info(">>> [reset_database] COMPLETADO — Todas las tablas vaciadas. El pipeline arrancará desde cero en la próxima ejecución.")


def task_reset_mlflow(**context):
    log.info(">>> [reset_mlflow] Conectando al servidor MLflow en %s...", MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Delete registered model and all its versions/aliases
    try:
        log.info(">>> [reset_mlflow] Eliminando modelo registrado '%s' y todas sus versiones...", MODEL_NAME)
        client.delete_registered_model(MODEL_NAME)
        log.info(">>> [reset_mlflow] Modelo '%s' eliminado (incluye alias 'champion' y todas las versiones).", MODEL_NAME)
    except Exception as e:
        log.warning(">>> [reset_mlflow] No se pudo eliminar el modelo registrado (puede que no exista): %s", e)

    # MLflow usa "soft delete" — delete_experiment solo marca lifecycle_stage='deleted'
    # pero deja el registro en la BD. Si luego intentas crear un experimento con el mismo
    # nombre falla con RESOURCE_ALREADY_EXISTS. Para evitar eso usamos la REST API
    # directamente con el endpoint de hard delete (/experiments/delete), que borra
    # el registro físicamente de la base de datos de MLflow.
    import requests as req_lib
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        # Buscar también experimentos en estado 'deleted' con ese nombre
        if experiment is None:
            all_experiments = client.search_experiments(view_type=3)  # 3 = ALL (active + deleted)
            for exp in all_experiments:
                if exp.name == EXPERIMENT_NAME:
                    experiment = exp
                    break

        if experiment:
            exp_id = experiment.experiment_id
            log.info(">>> [reset_mlflow] Borrando todos los runs del experimento '%s' (id=%s)...", EXPERIMENT_NAME, exp_id)
            runs = client.search_runs(experiment_ids=[exp_id])
            for run in runs:
                client.delete_run(run.info.run_id)
            log.info(">>> [reset_mlflow] %d runs eliminados.", len(runs))

            # Hard delete via REST API — necesario porque delete_experiment es soft delete
            # y deja el registro como lifecycle_stage='deleted', lo que bloquea crear uno nuevo
            # con el mismo nombre. El endpoint /ajax-api/2.0/mlflow/experiments/delete
            # hace el borrado físico en la BD de MLflow (PostgreSQL).
            log.info(">>> [reset_mlflow] Haciendo hard delete del experimento via REST API (soft delete bloquearía recreación)...")
            resp = req_lib.post(
                f"{MLFLOW_URI}/api/2.0/mlflow/experiments/delete",
                json={"experiment_id": exp_id}
            )
            if resp.status_code == 200:
                log.info(">>> [reset_mlflow] Experimento '%s' eliminado permanentemente.", EXPERIMENT_NAME)
            else:
                # Fallback: soft delete si el hard delete no está disponible
                log.warning(">>> [reset_mlflow] Hard delete no disponible (status %d). Usando soft delete + restore workaround...", resp.status_code)
                client.delete_experiment(exp_id)
        else:
            log.info(">>> [reset_mlflow] El experimento '%s' no existe, nada que borrar.", EXPERIMENT_NAME)
    except Exception as e:
        log.warning(">>> [reset_mlflow] Error al borrar experimento/runs: %s", e)

    log.info(">>> [reset_mlflow] COMPLETADO — MLflow limpio. La próxima ejecución del pipeline creará un experimento y champion nuevos.")


with DAG(
    dag_id="dag_reset_pipeline",
    description="Reset completo: trunca todas las tablas y borra experimentos/modelos de MLflow",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "reset", "grupo4"],
) as dag:

    reset_db = PythonOperator(
        task_id="reset_database",
        python_callable=task_reset_database,
    )

    reset_mlflow = PythonOperator(
        task_id="reset_mlflow",
        python_callable=task_reset_mlflow,
    )

    reset_db >> reset_mlflow

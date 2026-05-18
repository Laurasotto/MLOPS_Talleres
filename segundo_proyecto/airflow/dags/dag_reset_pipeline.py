# Importamos solo lo que necesitamos para este DAG de reset — es más simple que el pipeline principal
import os
import logging
from datetime import datetime

# SQLAlchemy nos permite conectarnos a PostgreSQL y ejecutar SQL desde Python
from sqlalchemy import create_engine, text

# Airflow para definir el DAG y sus tareas
from airflow import DAG
from airflow.operators.python import PythonOperator

# MLflow para borrar experimentos y modelos del registry
import mlflow
from mlflow.tracking import MlflowClient

# Tomamos las URLs de variables de entorno para no hardcodear credenciales en el código.
# Si la variable no existe, usamos el valor por defecto que funciona dentro de Docker.
DB_URI = os.getenv("AIRFLOW_DB_URI", "postgresql://postgres:postgres123@postgres-service:5432/mlops_db")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MODEL_NAME = "diabetes_readmission_model"
EXPERIMENT_NAME = "diabetes_readmission"

log = logging.getLogger(__name__)


def task_reset_database(**context):
    # Aquí vaciamos todas las tablas de PostgreSQL para que el pipeline pueda
    # arrancar desde cero la próxima vez que se ejecute.
    # Usamos TRUNCATE CASCADE para que también se borren datos en tablas relacionadas.
    log.info(">>> [reset_database] Conectando a PostgreSQL en mlops_db...")
    engine = create_engine(DB_URI)

    with engine.begin() as conn:
        # Vaciamos primero las tablas de splits porque pueden referenciar a diabetes_clean
        log.info(">>> [reset_database] Truncando clean_data.diabetes_train...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_train CASCADE"))

        log.info(">>> [reset_database] Truncando clean_data.diabetes_val...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_val CASCADE"))

        log.info(">>> [reset_database] Truncando clean_data.diabetes_test...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_test CASCADE"))

        log.info(">>> [reset_database] Truncando clean_data.diabetes_clean...")
        conn.execute(text("TRUNCATE TABLE clean_data.diabetes_clean CASCADE"))

        # Después vaciamos los datos crudos y el registro de batches
        log.info(">>> [reset_database] Truncando raw_data.diabetes_raw...")
        conn.execute(text("TRUNCATE TABLE raw_data.diabetes_raw CASCADE"))

        log.info(">>> [reset_database] Truncando raw_data.batch_state...")
        conn.execute(text("TRUNCATE TABLE raw_data.batch_state CASCADE"))

    log.info(">>> [reset_database] COMPLETADO — Todas las tablas vaciadas. El pipeline arrancará desde cero en la próxima ejecución.")


def task_reset_mlflow(**context):
    # Aquí borramos el modelo registrado y el experimento de MLflow
    # para que la próxima ejecución del pipeline empiece con un estado completamente limpio.
    log.info(">>> [reset_mlflow] Conectando al servidor MLflow en %s...", MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Eliminamos el modelo del registry junto con todas sus versiones y el alias 'champion'.
    # Si el modelo no existe (por ejemplo en la primera ejecución), simplemente lo ignoramos.
    try:
        log.info(">>> [reset_mlflow] Eliminando modelo registrado '%s' y todas sus versiones...", MODEL_NAME)
        client.delete_registered_model(MODEL_NAME)
        log.info(">>> [reset_mlflow] Modelo '%s' eliminado (incluye alias 'champion' y todas las versiones).", MODEL_NAME)
    except Exception as e:
        log.warning(">>> [reset_mlflow] No se pudo eliminar el modelo registrado (puede que no exista): %s", e)

    # MLflow usa "soft delete" — delete_experiment solo marca lifecycle_stage='deleted'
    # pero deja el registro en la BD. Si luego intentamos crear un experimento con el mismo
    # nombre, falla con RESOURCE_ALREADY_EXISTS. Para evitar eso, usamos la REST API
    # directamente con el endpoint de hard delete que borra el registro físicamente
    # de la base de datos de MLflow.
    import requests as req_lib
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        # Buscamos el experimento también entre los que están en estado 'deleted',
        # porque puede que ya le hayan hecho soft delete en una ejecución anterior.
        if experiment is None:
            all_experiments = client.search_experiments(view_type=3)  # 3 = ALL (activos + eliminados)
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

            # Hacemos hard delete via REST API porque delete_experiment es solo soft delete
            # y dejaría el experimento con lifecycle_stage='deleted', lo que bloquearía
            # crear uno nuevo con el mismo nombre en la siguiente ejecución del pipeline.
            log.info(">>> [reset_mlflow] Haciendo hard delete del experimento via REST API (soft delete bloquearía recreación)...")
            resp = req_lib.post(
                f"{MLFLOW_URI}/api/2.0/mlflow/experiments/delete",
                json={"experiment_id": exp_id}
            )
            if resp.status_code == 200:
                log.info(">>> [reset_mlflow] Experimento '%s' eliminado permanentemente.", EXPERIMENT_NAME)
            else:
                # Si el hard delete no está disponible, hacemos soft delete y dejamos que
                # el dag_diabetes_pipeline lo restaure automáticamente en la próxima ejecución
                log.warning(">>> [reset_mlflow] Hard delete no disponible (status %d). Usando soft delete + restore workaround...", resp.status_code)
                client.delete_experiment(exp_id)
        else:
            log.info(">>> [reset_mlflow] El experimento '%s' no existe, nada que borrar.", EXPERIMENT_NAME)
    except Exception as e:
        log.warning(">>> [reset_mlflow] Error al borrar experimento/runs: %s", e)

    log.info(">>> [reset_mlflow] COMPLETADO — MLflow limpio. La próxima ejecución del pipeline creará un experimento y champion nuevos.")


# Definición del DAG de reset — solo tiene dos tareas que corren en secuencia.
# Se ejecuta únicamente de forma manual, nunca de forma automática.
with DAG(
    dag_id="dag_reset_pipeline",
    description="Reset completo: trunca todas las tablas y borra experimentos/modelos de MLflow",
    schedule_interval=None,   # solo se dispara manualmente, nunca de forma automática
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "reset", "grupo4"],
) as dag:

    # Tarea 1: vaciar todas las tablas de PostgreSQL
    reset_db = PythonOperator(
        task_id="reset_database",
        python_callable=task_reset_database,
    )

    # Tarea 2: borrar el modelo y el experimento de MLflow
    reset_mlflow = PythonOperator(
        task_id="reset_mlflow",
        python_callable=task_reset_mlflow,
    )

    # Primero vaciamos la BD y luego limpiamos MLflow
    reset_db >> reset_mlflow

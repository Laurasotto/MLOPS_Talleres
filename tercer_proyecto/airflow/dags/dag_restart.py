from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import requests
import logging

DATA_API_URL = "http://host.docker.internal:8000"
GROUP_NUMBER = 4

default_args = {
    "owner": "grupo4",
    "retries": 1,
    "retry_delay": 10,
}

with DAG(
    dag_id="dag_reiniciar_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    @task
    def limpiar_base_de_datos():
        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            DROP TABLE IF EXISTS clean.properties  CASCADE;
            DROP TABLE IF EXISTS raw.batch_audit   CASCADE;
            DROP TABLE IF EXISTS raw.properties    CASCADE;
            DROP TABLE IF EXISTS raw.inference_events CASCADE;
            DROP TABLE IF EXISTS raw.batches       CASCADE;

            CREATE TABLE raw.batches (
                id           SERIAL PRIMARY KEY,
                batch_number INTEGER NOT NULL UNIQUE,
                fetched_at   TIMESTAMP NOT NULL DEFAULT now(),
                record_count INTEGER NOT NULL,
                fetch_status VARCHAR(20) NOT NULL
            );

            CREATE TABLE raw.properties (
                id             SERIAL PRIMARY KEY,
                batch_id       INTEGER NOT NULL REFERENCES raw.batches(id),
                brokered_by    DOUBLE PRECISION,
                status         VARCHAR(20),
                price          DOUBLE PRECISION,
                bed            DOUBLE PRECISION,
                bath           DOUBLE PRECISION,
                acre_lot       DOUBLE PRECISION,
                street         DOUBLE PRECISION,
                city           VARCHAR(100),
                state          VARCHAR(100),
                zip_code       VARCHAR(20),
                house_size     DOUBLE PRECISION,
                prev_sold_date VARCHAR(20),
                ingested_at    TIMESTAMP NOT NULL DEFAULT now()
            );

            CREATE TABLE raw.batch_audit (
                id               SERIAL PRIMARY KEY,
                batch_id         INTEGER NOT NULL REFERENCES raw.batches(id),
                dag_run_id       VARCHAR(200),
                executed_at      TIMESTAMP NOT NULL DEFAULT now(),
                schema_valid     BOOLEAN,
                quality_valid    BOOLEAN,
                drift_detected   BOOLEAN,
                new_categories   JSONB,
                volume_sufficient BOOLEAN,
                decision         VARCHAR(10),
                decision_reason  TEXT,
                candidate_run_id VARCHAR(200),
                candidate_mae    DOUBLE PRECISION,
                candidate_rmse   DOUBLE PRECISION,
                candidate_r2     DOUBLE PRECISION,
                promoted         BOOLEAN,
                promotion_reason TEXT
            );

            CREATE TABLE raw.inference_events (
                id              SERIAL PRIMARY KEY,
                requested_at    TIMESTAMP NOT NULL DEFAULT now(),
                input_data      JSONB,
                prediction      DOUBLE PRECISION,
                model_version   VARCHAR(100),
                model_alias     VARCHAR(100),
                response_status VARCHAR(20),
                error_message   TEXT
            );

            CREATE TABLE clean.properties (
                id             SERIAL PRIMARY KEY,
                batch_id       INTEGER NOT NULL REFERENCES raw.batches(id),
                price          DOUBLE PRECISION NOT NULL,
                bed            DOUBLE PRECISION,
                bath           DOUBLE PRECISION,
                acre_lot       DOUBLE PRECISION,
                house_size     DOUBLE PRECISION,
                status         VARCHAR(20),
                city           VARCHAR(100),
                state          VARCHAR(100),
                zip_code       VARCHAR(20),
                prev_sold_date VARCHAR(20),
                processed_at   TIMESTAMP NOT NULL DEFAULT now()
            );
        """)

        conn.commit()
        conn.close()
        logging.info("Base de datos limpia y tablas recreadas.")

    @task
    def resetear_api():
        response = requests.get(
            f"{DATA_API_URL}/restart_data_generation",
            params={"group_number": GROUP_NUMBER},
            timeout=30,
        )
        response.raise_for_status()
        logging.info(f"API reseteada: {response.json()}")

    # Ambas tareas corren en paralelo — son independientes entre sí
    limpiar_base_de_datos()
    resetear_api()

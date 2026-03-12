from datetime import datetime
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


POSTGRES_CONN_ID = "postgres_default"
TABLE_NAME = "training_data"
API_BASE_URL = "http://host.docker.internal:8080/data"
GROUP_NUMBER = 4


def create_table():
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    sql = f"""
    CREATE TABLE IF NOT EXISTS public.{TABLE_NAME} (
        id                                      SERIAL PRIMARY KEY,
        group_number                            INTEGER,
        batch_number                            INTEGER,
        elevation                               INTEGER,
        aspect                                  INTEGER,
        slope                                   INTEGER,
        horizontal_distance_to_hydrology        INTEGER,
        vertical_distance_to_hydrology          INTEGER,
        horizontal_distance_to_roadways         INTEGER,
        hillshade_9am                           INTEGER,
        hillshade_noon                          INTEGER,
        hillshade_3pm                           INTEGER,
        horizontal_distance_to_fire_points      INTEGER,
        wilderness_area                         VARCHAR(100),
        soil_type                               VARCHAR(100),
        cover_type                              INTEGER,
        inserted_at                             TIMESTAMP DEFAULT NOW()
    );
    """

    print("Creando/verificando tabla en Postgres...")
    hook.run(sql)
    print("Tabla lista")


def get_api_data():
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur = conn.cursor()

    insert_sql = f"""
    INSERT INTO public.{TABLE_NAME} (
        group_number,
        batch_number,
        elevation,
        aspect,
        slope,
        horizontal_distance_to_hydrology,
        vertical_distance_to_hydrology,
        horizontal_distance_to_roadways,
        hillshade_9am,
        hillshade_noon,
        hillshade_3pm,
        horizontal_distance_to_fire_points,
        wilderness_area,
        soil_type,
        cover_type
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    """

    try:
        params = {
            "group_number": GROUP_NUMBER
        }

        print(f"Llamando API: {API_BASE_URL}")
        print(f"Parametros enviados: {params}")

        response = requests.get(API_BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()

        real_group_number = int(payload["group_number"])
        real_batch_number = int(payload["batch_number"])
        rows = payload["data"]

        print(
            f"Payload recibido -> group_number={real_group_number}, "
            f"batch_number={real_batch_number}, filas={len(rows)}"
        )

        for r in rows:
            print(f"Insertando fila: {r}")

            cur.execute(
                insert_sql,
                (
                    real_group_number,
                    real_batch_number,
                    int(r[0]),
                    int(r[1]),
                    int(r[2]),
                    int(r[3]),
                    int(r[4]),
                    int(r[5]),
                    int(r[6]),
                    int(r[7]),
                    int(r[8]),
                    int(r[9]),
                    str(r[10]),
                    str(r[11]),
                    int(r[12])
                )
            )

        conn.commit()
        print("Insert completado")

        cur.execute(f"SELECT COUNT(*) FROM public.{TABLE_NAME}")
        total_rows = cur.fetchone()[0]
        print(f"COUNT actual en public.{TABLE_NAME}: {total_rows}")

    finally:
        cur.close()
        conn.close()
        print("Conexion cerrada")


with DAG(
    dag_id="dag_postgres_single_api_test",
    start_date=datetime(2026, 3, 11),
    schedule=None,
    catchup=False,
    tags=["mlops", "postgres", "api", "single"]
) as dag:

    task_create_table = PythonOperator(
        task_id="create_table",
        python_callable=create_table
    )

    task_get_api_data = PythonOperator(
        task_id="get_api_data",
        python_callable=get_api_data
    )

    task_create_table >> task_get_api_data

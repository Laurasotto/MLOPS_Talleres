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
    "retries": 3,
    "retry_delay": 30,
}

with DAG(
    dag_id="dag_mlops_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    @task
    def fetch_batch_from_api():
        try:
            # Llama al endpoint /data de la API del profesor pasando el numero de grupo
            response = requests.get(
                f"{DATA_API_URL}/data",
                params={"group_number": GROUP_NUMBER},
                timeout=30,  # si la API no responde en 30 segundos, falla
            )

            # Cuando se acaban los batches la API devuelve 400 con mensaje de fin
            # En ese caso no es un error real, simplemente no hay mas datos
            if response.status_code == 400:
                logging.info("No hay mas batches disponibles.")
                return None

            # Si hay otro tipo de error (500, 503, etc.) si lanzamos excepcion
            response.raise_for_status()

            # Convertimos la respuesta JSON a un diccionario de Python
            data = response.json()
            logging.info(f"Batch {data['batch_number']} obtenido: {len(data['data'])} registros.")
            return data

        except requests.exceptions.RequestException as e:
            # Cualquier error de red (timeout, conexion rechazada, etc.) llega aqui
            # RuntimeError hace que Airflow marque la tarea como fallida y reintente
            raise RuntimeError(f"Error al llamar la API: {e}")

    @task
    def store_raw_batch(data: dict):
        # Si fetch_batch_from_api devolvio None significa que no habia mas batches
        if data is None:
            logging.info("No hay datos que almacenar.")
            return None

        # PostgresHook es la forma que tiene Airflow de conectarse a PostgreSQL
        # postgres_mlops es el nombre de la conexion configurada en la UI de Airflow
        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()

        # Insertamos el registro del batch en raw.batches
        # ON CONFLICT DO NOTHING evita duplicados si el DAG se corre dos veces con el mismo batch
        # RETURNING id nos devuelve el id asignado automaticamente por la BD
        cursor.execute(
            """
            INSERT INTO raw.batches (batch_number, record_count, fetch_status)
            VALUES (%s, %s, %s)
            ON CONFLICT (batch_number) DO NOTHING
            RETURNING id
            """,
            (data["batch_number"], len(data["data"]), "success"),
        )
        row = cursor.fetchone()

        # Si row es None significa que el batch ya existia (el ON CONFLICT lo ignoro)
        # No insertamos los registros de nuevo para no duplicar datos
        if row is None:
            logging.warning(f"Batch {data['batch_number']} ya existia en la base de datos.")
            conn.close()
            return None

        # El id del batch recien insertado, lo usamos como llave foranea en raw.properties
        batch_id = row[0]

        # Convertimos la lista de registros del batch en una lista de tuplas
        # Cada tupla tiene los valores en el mismo orden que las columnas del INSERT de abajo
        records = [
            (
                batch_id,
                r.get("brokered_by"),
                r.get("status"),
                r.get("price"),
                r.get("bed"),
                r.get("bath"),
                r.get("acre_lot"),
                r.get("street"),
                r.get("city"),
                r.get("state"),
                r.get("zip_code"),
                r.get("house_size"),
                r.get("prev_sold_date"),
            )
            for r in data["data"]  # esto recorre cada propiedad del batch
        ]

        # executemany inserta todas las filas de una sola vez, es mucho mas eficiente
        # que hacer un INSERT por cada registro (el batch puede tener 300k registros)
        cursor.executemany(
            """
            INSERT INTO raw.properties
                (batch_id, brokered_by, status, price, bed, bath,
                 acre_lot, street, city, state, zip_code, house_size, prev_sold_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )

        # commit guarda todos los cambios en la BD, sin esto no se persiste nada
        conn.commit()
        conn.close()
        logging.info(f"Batch {data['batch_number']} almacenado con batch_id={batch_id}.")

        # Retornamos el batch_id para que la siguiente tarea sepa que batch procesar
        return batch_id

    # =========================================================================
    # VALIDACION DE ESQUEMA
    # Verifica que el batch tenga las columnas correctas y los tipos esperados
    # =========================================================================

    @task
    def validate_schema(data: dict):
        # Si no hay datos no hay nada que validar
        if data is None:
            return {"valid": False, "reason": "sin datos"}

        # Columnas que esperamos recibir siempre en cada batch
        expected_columns = {
            "brokered_by", "status", "price", "bed", "bath",
            "acre_lot", "street", "city", "state", "zip_code",
            "house_size", "prev_sold_date",
        }

        # Tomamos las columnas que trae el primer registro del batch
        received_columns = set(data["data"][0].keys())

        # Comparamos columnas faltantes y columnas nuevas inesperadas
        missing = expected_columns - received_columns
        extra = received_columns - expected_columns

        if missing:
            logging.error(f"Columnas faltantes en el batch: {missing}")
            return {"valid": False, "reason": f"columnas faltantes: {missing}"}

        if extra:
            # Columnas nuevas no rompen el pipeline pero las registramos como advertencia
            logging.warning(f"Columnas nuevas no esperadas: {extra}")

        logging.info("Esquema validado correctamente.")
        return {"valid": True, "extra_columns": list(extra)}

    # =========================================================================
    # VALIDACION DE CALIDAD
    # Verifica nulos, duplicados y que el target (price) tenga valores validos
    # =========================================================================

    @task
    def validate_data_quality(batch_id: int, data: dict):
        if data is None or batch_id is None:
            return {"valid": False, "reason": "sin datos"}

        records = data["data"]
        total = len(records)

        # Contamos cuantos registros tienen price nulo o igual a cero
        # price es el target, un registro sin precio no sirve para entrenar
        invalid_price = sum(1 for r in records if not r.get("price") or r["price"] <= 0)

        # Contamos nulos en las columnas numericas clave
        nulls = {
            col: sum(1 for r in records if r.get(col) is None)
            for col in ["bed", "bath", "acre_lot", "house_size"]
        }

        # Un batch con mas del 20% de precios invalidos no es util para entrenar
        price_invalid_pct = invalid_price / total
        if price_invalid_pct > 0.20:
            logging.error(f"{price_invalid_pct:.1%} de registros tienen price invalido.")
            return {"valid": False, "reason": f"demasiados precios invalidos: {price_invalid_pct:.1%}"}

        logging.info(f"Calidad validada. Nulos por columna: {nulls}")
        return {
            "valid": True,
            "total_records": total,
            "invalid_price_count": invalid_price,
            "nulls_per_column": nulls,
        }

    batch_data = fetch_batch_from_api()
    batch_id   = store_raw_batch(batch_data)
    schema_ok  = validate_schema(batch_data)
    quality_ok = validate_data_quality(batch_id, batch_data)

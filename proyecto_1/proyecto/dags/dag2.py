from datetime import datetime
import json
import time
import uuid
import requests
import websocket
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ──────────────────────────────────────────────────────────────────────────────
POSTGRES_CONN_ID  = "postgres_default"
TABLE_NAME        = "training_data"
PROCESSED_TABLE   = "training_data_processed"
API_BASE_URL      = "http://host.docker.internal:8080/data"
GROUP_NUMBER      = 4

JUPYTER_BASE_URL  = "http://jupyter:8888"
JUPYTER_WS_URL    = "ws://jupyter:8888"
JUPYTER_TOKEN     = "mlops_token"
NOTEBOOK_NAME     = "train_model.ipynb"

MINIO_ENDPOINT    = "http://minio:9000"
MINIO_ACCESS_KEY  = "minioadmin"
MINIO_SECRET_KEY  = "minioadmin"
MINIO_BUCKET      = "models"


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 1: create_tables
# Borra y recrea las tablas en cada ejecución para empezar desde cero.
# ══════════════════════════════════════════════════════════════════════════════
def create_tables():
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    print("Eliminando tablas anteriores si existen...")
    hook.run(f"DROP TABLE IF EXISTS public.{PROCESSED_TABLE};")
    hook.run(f"DROP TABLE IF EXISTS public.{TABLE_NAME};")
    print("Tablas eliminadas")

    raw_sql = f"""
    CREATE TABLE public.{TABLE_NAME} (
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

    processed_sql = f"""
    CREATE TABLE public.{PROCESSED_TABLE} (
        id                                      SERIAL PRIMARY KEY,
        group_number                            INTEGER,
        batch_number                            INTEGER,
        elevation                               FLOAT,
        aspect                                  FLOAT,
        slope                                   FLOAT,
        horizontal_distance_to_hydrology        FLOAT,
        vertical_distance_to_hydrology          FLOAT,
        horizontal_distance_to_roadways         FLOAT,
        hillshade_9am                           FLOAT,
        hillshade_noon                          FLOAT,
        hillshade_3pm                           FLOAT,
        horizontal_distance_to_fire_points      FLOAT,
        wilderness_area_encoded                 INTEGER,
        soil_type_encoded                       INTEGER,
        cover_type                              INTEGER,
        inserted_at                             TIMESTAMP DEFAULT NOW()
    );
    """

    print("Creando tablas nuevas...")
    hook.run(raw_sql)
    hook.run(processed_sql)
    print("Tablas creadas y listas")


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 2: get_api_data
# Llama a la API externa y guarda los datos crudos en PostgreSQL.
# ══════════════════════════════════════════════════════════════════════════════
def get_api_data(**context):
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur  = conn.cursor()

    insert_sql = f"""
    INSERT INTO public.{TABLE_NAME} (
        group_number, batch_number,
        elevation, aspect, slope,
        horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
        horizontal_distance_to_roadways,
        hillshade_9am, hillshade_noon, hillshade_3pm,
        horizontal_distance_to_fire_points,
        wilderness_area, soil_type, cover_type
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    """

    try:
        print(f"Llamando API: {API_BASE_URL} | group_number={GROUP_NUMBER}")
        response = requests.get(
            API_BASE_URL,
            params={"group_number": GROUP_NUMBER},
            timeout=30
        )
        response.raise_for_status()

        payload           = response.json()
        real_group_number = int(payload["group_number"])
        real_batch_number = int(payload["batch_number"])
        rows              = payload["data"]

        print(f"Recibido: group={real_group_number}, batch={real_batch_number}, filas={len(rows)}")

        for r in rows:
            cur.execute(insert_sql, (
                real_group_number, real_batch_number,
                int(r[0]), int(r[1]), int(r[2]),
                int(r[3]), int(r[4]), int(r[5]),
                int(r[6]), int(r[7]), int(r[8]),
                int(r[9]),
                str(r[10]), str(r[11]),
                int(r[12])
            ))

        conn.commit()
        print("Insercion completada")

        cur.execute(f"SELECT COUNT(*) FROM public.{TABLE_NAME}")
        print(f"Total filas en {TABLE_NAME}: {cur.fetchone()[0]}")

        context["ti"].xcom_push(key="batch_number", value=real_batch_number)
        context["ti"].xcom_push(key="group_number", value=real_group_number)

    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 3: preprocess_data
# Lee el batch recien insertado, aplica Label Encoding y normalizacion Min-Max.
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_data(**context):
    batch_number = context["ti"].xcom_pull(key="batch_number", task_ids="get_api_data")
    group_number = context["ti"].xcom_pull(key="group_number", task_ids="get_api_data")

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = hook.get_conn()
    cur  = conn.cursor()

    try:
        cur.execute(f"""
            SELECT
                group_number, batch_number,
                elevation, aspect, slope,
                horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
                horizontal_distance_to_roadways,
                hillshade_9am, hillshade_noon, hillshade_3pm,
                horizontal_distance_to_fire_points,
                wilderness_area, soil_type, cover_type
            FROM public.{TABLE_NAME}
            WHERE batch_number = %s AND group_number = %s
        """, (batch_number, group_number))

        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No hay filas para batch={batch_number}, group={group_number}")

        columns = [
            "group_number", "batch_number",
            "elevation", "aspect", "slope",
            "horizontal_distance_to_hydrology", "vertical_distance_to_hydrology",
            "horizontal_distance_to_roadways",
            "hillshade_9am", "hillshade_noon", "hillshade_3pm",
            "horizontal_distance_to_fire_points",
            "wilderness_area", "soil_type", "cover_type"
        ]
        df = pd.DataFrame(rows, columns=columns)
        print(f"Filas a procesar: {len(df)}")

        le_wilderness = LabelEncoder()
        le_soil       = LabelEncoder()
        df["wilderness_area_encoded"] = le_wilderness.fit_transform(df["wilderness_area"])
        df["soil_type_encoded"]       = le_soil.fit_transform(df["soil_type"])

        numeric_cols = [
            "elevation", "aspect", "slope",
            "horizontal_distance_to_hydrology", "vertical_distance_to_hydrology",
            "horizontal_distance_to_roadways",
            "hillshade_9am", "hillshade_noon", "hillshade_3pm",
            "horizontal_distance_to_fire_points"
        ]
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            df[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

        insert_processed = f"""
        INSERT INTO public.{PROCESSED_TABLE} (
            group_number, batch_number,
            elevation, aspect, slope,
            horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
            horizontal_distance_to_roadways,
            hillshade_9am, hillshade_noon, hillshade_3pm,
            horizontal_distance_to_fire_points,
            wilderness_area_encoded, soil_type_encoded, cover_type
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        """

        for _, row in df.iterrows():
            cur.execute(insert_processed, (
                int(row["group_number"]), int(row["batch_number"]),
                float(row["elevation"]), float(row["aspect"]), float(row["slope"]),
                float(row["horizontal_distance_to_hydrology"]),
                float(row["vertical_distance_to_hydrology"]),
                float(row["horizontal_distance_to_roadways"]),
                float(row["hillshade_9am"]), float(row["hillshade_noon"]),
                float(row["hillshade_3pm"]),
                float(row["horizontal_distance_to_fire_points"]),
                int(row["wilderness_area_encoded"]),
                int(row["soil_type_encoded"]),
                int(row["cover_type"])
            ))

        conn.commit()
        print(f"Preprocesamiento completado: {len(df)} filas en {PROCESSED_TABLE}")

    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 4: trigger_jupyter
# Ejecuta el notebook train_model.ipynb en Jupyter via WebSocket.
#
# Flujo:
#   A) Crear kernel via HTTP
#   B) Abrir conexion WebSocket al canal del kernel
#   C) Esperar a que el kernel este idle
#   D) Leer el notebook
#   E) Inyectar parametros como celda separada
#   F) Ejecutar cada celda individualmente
#      CAMBIO: En vez de concatenar todas las celdas en un solo string
#      (causaba SyntaxError por caracteres especiales en los comentarios),
#      ahora se ejecuta cada celda por separado esperando que termine
#      antes de enviar la siguiente. Ademas, se maneja correctamente
#      el caso en que source es lista o string.
#   G) Cerrar WebSocket y eliminar kernel
# ══════════════════════════════════════════════════════════════════════════════
def trigger_jupyter(**context):
    batch_number = context["ti"].xcom_pull(key="batch_number", task_ids="get_api_data")
    group_number = context["ti"].xcom_pull(key="group_number", task_ids="get_api_data")

    headers = {"Authorization": f"token {JUPYTER_TOKEN}"}

    def execute_cell(ws, code, cell_num):
        """Envia una celda al kernel via WebSocket y espera que termine."""
        msg_id = str(uuid.uuid4())
        msg = {
            "header": {
                "msg_id":   msg_id,
                "username": "airflow",
                "session":  str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version":  "5.0"
            },
            "parent_header": {},
            "metadata":      {},
            "content": {
                "code":             code,
                "silent":           False,
                "store_history":    True,
                "user_expressions": {},
                "allow_stdin":      False
            }
        }
        ws.send(json.dumps(msg))

        errors   = []
        max_time = 300
        start    = time.time()

        while time.time() - start < max_time:
            try:
                ws.settimeout(10)
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            m        = json.loads(raw)
            msg_type = m.get("msg_type", "")
            content  = m.get("content", {})

            if msg_type == "stream":
                print(f"  [celda {cell_num}] {content.get('text', '').strip()}")
            elif msg_type == "error":
                errors.append(f"{content.get('ename')}: {content.get('evalue')}")
                print(f"  [celda {cell_num} ERROR] {content.get('ename')}: {content.get('evalue')}")
            elif msg_type == "execute_reply":
                status = content.get("status", "")
                if status == "error":
                    errors.append(f"execute_reply: {content.get('ename')}: {content.get('evalue')}")
                return errors

        return [f"Celda {cell_num} supero el timeout de {max_time}s"]

    # Paso A: Crear kernel
    print("Creando kernel en Jupyter...")
    r = requests.post(
        f"{JUPYTER_BASE_URL}/api/kernels",
        headers=headers,
        json={"name": "python3"}
    )
    r.raise_for_status()
    kernel_id = r.json()["id"]
    print(f"Kernel creado: {kernel_id}")

    try:
        # Paso B: Abrir WebSocket
        ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels?token={JUPYTER_TOKEN}"
        print("Conectando WebSocket...")
        ws = websocket.create_connection(ws_url, timeout=30)
        print("WebSocket conectado")

        # Paso C: Esperar idle
        print("Esperando que el kernel este listo...")
        for _ in range(30):
            msg = json.loads(ws.recv())
            if msg.get("msg_type") == "status":
                state = msg.get("content", {}).get("execution_state", "")
                print(f"  Estado kernel: {state}")
                if state == "idle":
                    print("Kernel listo")
                    break
            time.sleep(1)

        # Paso D: Leer el notebook
        print(f"Leyendo notebook {NOTEBOOK_NAME}...")
        nb_response = requests.get(
            f"{JUPYTER_BASE_URL}/api/contents/{NOTEBOOK_NAME}",
            headers=headers
        )
        nb_response.raise_for_status()
        cells = nb_response.json()["content"]["cells"]

        # Paso E: Inyectar parametros como celda separada
        param_code = (
            "import os\n"
            f"os.environ['BATCH_NUMBER'] = '{batch_number}'\n"
            f"os.environ['GROUP_NUMBER'] = '{group_number}'\n"
            f"print('Parametros inyectados: batch={batch_number}, group={group_number}')\n"
        )
        print("Inyectando parametros...")
        errors = execute_cell(ws, param_code, cell_num=0)
        if errors:
            raise RuntimeError(f"Error inyectando parametros: {errors}")

        # Paso F: Ejecutar cada celda individualmente
        # CAMBIO: Se ejecuta celda por celda en vez de concatenar todo en un
        # solo string. Esto evita el SyntaxError por caracteres especiales en
        # los comentarios del notebook. Ademas, cell["source"] puede ser string
        # o lista de strings, por eso se usa isinstance para manejarlo.
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        print(f"Ejecutando {len(code_cells)} celdas del notebook...")

        for i, cell in enumerate(code_cells, start=1):
            # CAMBIO: manejar source como string o lista de strings
            src = cell["source"]
            cell_code = src if isinstance(src, str) else "".join(src)

            if not cell_code.strip():
                continue

            print(f"  Ejecutando celda {i}/{len(code_cells)}...")
            errors = execute_cell(ws, cell_code, cell_num=i)
            if errors:
                raise RuntimeError(f"Error en celda {i}:\n" + "\n".join(errors))

        print("Todas las celdas ejecutadas exitosamente")

    finally:
        # Paso G: Cerrar WebSocket y eliminar kernel
        try:
            ws.close()
        except Exception:
            pass
        requests.delete(
            f"{JUPYTER_BASE_URL}/api/kernels/{kernel_id}",
            headers=headers
        )
        print(f"Kernel {kernel_id} eliminado")


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 5: verify_model_saved
# Verifica que el notebook guardo el modelo en MinIO correctamente.
# ══════════════════════════════════════════════════════════════════════════════
def verify_model_saved(**context):
    import boto3
    from botocore.client import Config

    batch_number = context["ti"].xcom_pull(key="batch_number", task_ids="get_api_data")
    group_number = context["ti"].xcom_pull(key="group_number", task_ids="get_api_data")

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

    prefix = f"group_{group_number}/"
    print(f"Buscando modelos en MinIO: bucket={MINIO_BUCKET}, prefix={prefix}")

    try:
        response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
    except Exception as e:
        raise RuntimeError(f"No se pudo conectar a MinIO o el bucket no existe: {e}")

    objects = response.get("Contents", [])
    if not objects:
        raise RuntimeError(
            f"No se encontro ningun modelo en MinIO para group={group_number}."
        )

    batch_models = [o for o in objects if f"batch_{batch_number}" in o["Key"]]
    if not batch_models:
        raise RuntimeError(
            f"No se encontro modelo para batch={batch_number} en MinIO. "
            f"Modelos existentes: {[o['Key'] for o in objects]}"
        )

    latest_model = batch_models[-1]
    print(f"Modelo verificado en MinIO:")
    print(f"  Ruta:   {latest_model['Key']}")
    print(f"  Tamano: {latest_model['Size']} bytes")
    print(f"  Fecha:  {latest_model['LastModified']}")
    print("Pipeline completado exitosamente.")


# ══════════════════════════════════════════════════════════════════════════════
# DEFINICION DEL DAG
# ══════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id="dag_mlops_full_pipeline",
    start_date=datetime(2026, 3, 11),
    schedule=None,
    catchup=False,
    tags=["mlops", "postgres", "minio", "jupyter", "random_forest"]
) as dag:

    task_create_tables = PythonOperator(
        task_id="create_tables",
        python_callable=create_tables
    )

    task_get_api_data = PythonOperator(
        task_id="get_api_data",
        python_callable=get_api_data,
        provide_context=True
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        provide_context=True
    )

    task_trigger_jupyter = PythonOperator(
        task_id="trigger_jupyter",
        python_callable=trigger_jupyter,
        provide_context=True
    )

    task_verify = PythonOperator(
        task_id="verify_model_saved",
        python_callable=verify_model_saved,
        provide_context=True
    )

    task_create_tables >> task_get_api_data >> task_preprocess >> task_trigger_jupyter >> task_verify

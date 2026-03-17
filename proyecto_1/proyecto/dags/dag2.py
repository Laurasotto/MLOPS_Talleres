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
from airflow.models import Variable, DagModel
from airflow import settings


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION GENERAL
# ──────────────────────────────────────────────────────────────────────────────
POSTGRES_CONN_ID  = "postgres_default"
TABLE_NAME        = "training_data"
PROCESSED_TABLE   = "training_data_processed"
READY_TABLE       = "training_data_ready"
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

# Numero maximo de ejecuciones antes de pausar el DAG.
# Variable de Airflow usada como contador: "pipeline_run_count"
# Se inicializa en 0 y se incrementa en 1 en cada ejecucion exitosa.
MAX_RUNS          = 10
COUNTER_VAR       = "pipeline_run_count"


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 1: create_tables
# Crea las tres tablas SI NO EXISTEN.
# A diferencia de antes, ya NO hace DROP — eso lo maneja dag_restart_data.
# Esto permite acumular datos de los 10 batches en las mismas tablas.
# ══════════════════════════════════════════════════════════════════════════════
def create_tables():
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    # CREATE TABLE IF NOT EXISTS: si la tabla ya existe, no hace nada.
    # Si no existe, la crea. Asi los datos se acumulan entre ejecuciones.
    raw_sql = f"""
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

    processed_sql = f"""
    CREATE TABLE IF NOT EXISTS public.{PROCESSED_TABLE} (
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

    ready_sql = f"""
    CREATE TABLE IF NOT EXISTS public.{READY_TABLE} (
        id                                      SERIAL PRIMARY KEY,
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
        cover_type                              INTEGER
    );
    """

    print("Verificando/creando tablas en Postgres...")
    hook.run(raw_sql)
    hook.run(processed_sql)
    hook.run(ready_sql)
    print("Tablas listas")


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
# Transforma los datos y los guarda en las dos tablas procesadas.
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

        insert_ready = f"""
        INSERT INTO public.{READY_TABLE} (
            elevation, aspect, slope,
            horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
            horizontal_distance_to_roadways,
            hillshade_9am, hillshade_noon, hillshade_3pm,
            horizontal_distance_to_fire_points,
            wilderness_area_encoded, soil_type_encoded, cover_type
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s
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

            cur.execute(insert_ready, (
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
        print(f"Preprocesamiento completado: {len(df)} filas en ambas tablas")

    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAREA 4: trigger_jupyter
# Ejecuta el notebook train_model.ipynb en Jupyter via WebSocket.
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
        ws_url = f"{JUPYTER_WS_URL}/api/kernels/{kernel_id}/channels?token={JUPYTER_TOKEN}"
        print("Conectando WebSocket...")
        ws = websocket.create_connection(ws_url, timeout=30)
        print("WebSocket conectado")

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

        print(f"Leyendo notebook {NOTEBOOK_NAME}...")
        nb_response = requests.get(
            f"{JUPYTER_BASE_URL}/api/contents/{NOTEBOOK_NAME}",
            headers=headers
        )
        nb_response.raise_for_status()
        cells = nb_response.json()["content"]["cells"]

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

        code_cells = [c for c in cells if c["cell_type"] == "code"]
        print(f"Ejecutando {len(code_cells)} celdas del notebook...")

        for i, cell in enumerate(code_cells, start=1):
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
# TAREA 5: verify_and_stop
# Incrementa el contador de ejecuciones usando Airflow Variables.
# Cuando llega a MAX_RUNS, pausa el DAG y resetea el contador a 0.
#
# Airflow Variables es un almacen clave-valor dentro de Airflow.
# Es la forma correcta de compartir estado entre ejecuciones del DAG.
# A diferencia de XCom (que es por ejecucion), las Variables persisten
# entre ejecuciones hasta que se borren explicitamente.
# ══════════════════════════════════════════════════════════════════════════════
def verify_and_stop(**context):
    import boto3
    from botocore.client import Config

    batch_number = context["ti"].xcom_pull(key="batch_number", task_ids="get_api_data")
    group_number = context["ti"].xcom_pull(key="group_number", task_ids="get_api_data")
    dag_id       = context["dag"].dag_id

    # Verificar que el modelo fue guardado en MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

    prefix = f"group_{group_number}/"
    response = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
    objects  = response.get("Contents", [])

    batch_models = [o for o in objects if f"batch_{batch_number}" in o["Key"]]
    if not batch_models:
        raise RuntimeError(f"No se encontro modelo para batch={batch_number} en MinIO.")

    latest_model = batch_models[-1]
    print(f"Modelo verificado: {latest_model['Key']} ({latest_model['Size']} bytes)")

    # ── Contador con Airflow Variables ────────────────────────────────────
    # Variable.get: lee el valor actual del contador.
    #   - Si no existe todavia, usa el default_var=0
    # Variable.set: guarda el nuevo valor del contador
    current_count = int(Variable.get(COUNTER_VAR, default_var=0))
    current_count += 1
    Variable.set(COUNTER_VAR, current_count)

    print(f"Ejecucion {current_count}/{MAX_RUNS} completada")

    if current_count >= MAX_RUNS:
        # Resetear el contador a 0 para la proxima ronda
        Variable.set(COUNTER_VAR, 0)
        print(f"Se completaron {MAX_RUNS} ejecuciones. Pausando el DAG...")

        # Pausar el DAG modificando su estado en la base de datos de Airflow
        session   = settings.Session()
        dag_model = session.query(DagModel).filter(DagModel.dag_id == dag_id).first()
        if dag_model:
            dag_model.is_paused = True
            session.commit()
        session.close()

        print(f"DAG '{dag_id}' pausado. Para una nueva ronda corre dag_restart_data y activa el toggle.")
    else:
        print(f"Faltan {MAX_RUNS - current_count} ejecuciones. El DAG seguira corriendo.")


# ══════════════════════════════════════════════════════════════════════════════
# DEFINICION DEL DAG
#
# schedule="*/6 * * * *": ejecutar cada minuto (crontab)
# max_active_runs=1:    solo una ejecucion a la vez
# catchup=False:        no correr ejecuciones pasadas al activar
# ══════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id="dag_mlops_full_pipeline",
    start_date=datetime(2026, 3, 11),
    schedule="*/6 * * * *",   # cada minuto
    max_active_runs=1,
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
        task_id="verify_and_stop",
        python_callable=verify_and_stop,
        provide_context=True
    )

    task_create_tables >> task_get_api_data >> task_preprocess >> task_trigger_jupyter >> task_verify

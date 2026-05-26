from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import requests
import logging
import os

DATA_API_URL    = "http://host.docker.internal:8000"
GROUP_NUMBER    = 4
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow.svc.cluster.local:5000")
EXPERIMENT_NAME = "precio_propiedades_grupo4"
PROD_ALIAS      = "production"

# Umbrales de decision de entrenamiento
MIN_RECORDS_TO_TRAIN    = 500   # minimo de registros limpios acumulados para entrenar
DRIFT_ALPHA             = 0.05  # p-value del KS test; si p < alpha hay drift
MIN_CATEGORY_FREQ       = 10    # categoria nueva debe aparecer al menos N veces para contar
MAE_IMPROVE_PCT         = 0.03  # candidato debe mejorar MAE del productivo en al menos 3%
MAX_RMSE_DEGRADE_PCT    = 0.01  # candidato no puede empeorar RMSE mas del 1%

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

    # =========================================================================
    # 1. OBTENER LOTE DE LA API
    # =========================================================================

    @task
    def obtener_lote_api():
        logging.info(f"Consultando API en {DATA_API_URL}/data para el grupo {GROUP_NUMBER}...")
        try:
            response = requests.get(
                f"{DATA_API_URL}/data",
                params={"group_number": GROUP_NUMBER},
                timeout=30,
            )
            if response.status_code == 400:
                logging.info("La API respondio 400: no hay mas batches disponibles para este grupo.")
                return None
            response.raise_for_status()
            data = response.json()
            logging.info(
                f"Batch {data['batch_number']} recibido exitosamente. "
                f"Total de registros: {len(data['data'])}."
            )
            return data
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error de red al llamar la API: {e}")

    # =========================================================================
    # 2. GUARDAR DATOS CRUDOS EN POSTGRES
    # =========================================================================

    @task
    def guardar_datos_postgres(data: dict):
        if data is None:
            logging.info("No hay datos que almacenar porque la tarea anterior no trajo ningun batch.")
            return None

        logging.info(f"Conectando a PostgreSQL para almacenar el batch {data['batch_number']}...")
        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()

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

        if row is None:
            logging.warning(
                f"El batch {data['batch_number']} ya existia en raw.batches. "
                f"Se omite la insercion para evitar duplicados."
            )
            conn.close()
            return None

        batch_id = row[0]
        logging.info(
            f"Batch registrado con batch_id={batch_id}. "
            f"Insertando {len(data['data'])} propiedades en raw.properties..."
        )

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
            for r in data["data"]
        ]

        cursor.executemany(
            """
            INSERT INTO raw.properties
                (batch_id, brokered_by, status, price, bed, bath,
                 acre_lot, street, city, state, zip_code, house_size, prev_sold_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )

        conn.commit()
        conn.close()
        logging.info(
            f"Listo. {len(records)} propiedades del batch {data['batch_number']} "
            f"guardadas en raw.properties con batch_id={batch_id}."
        )
        return batch_id

    # =========================================================================
    # 3. VALIDAR COLUMNAS (esquema)
    # =========================================================================

    @task
    def validar_columnas(data: dict):
        if data is None:
            logging.info("No hay datos que validar porque el batch vino vacio.")
            return {"valid": False, "reason": "sin datos"}

        expected_columns = {
            "brokered_by", "status", "price", "bed", "bath",
            "acre_lot", "street", "city", "state", "zip_code",
            "house_size", "prev_sold_date",
        }
        received_columns = set(data["data"][0].keys())

        logging.info(f"Columnas esperadas:  {sorted(expected_columns)}")
        logging.info(f"Columnas recibidas:  {sorted(received_columns)}")

        missing = expected_columns - received_columns
        extra   = received_columns - expected_columns

        if missing:
            logging.error(
                f"Faltan {len(missing)} columnas obligatorias: {missing}. "
                f"El batch no puede procesarse."
            )
            return {"valid": False, "reason": f"columnas faltantes: {missing}"}

        if extra:
            logging.warning(
                f"El batch trae {len(extra)} columnas nuevas no esperadas: {extra}. "
                f"Se ignoraran pero conviene revisarlas."
            )

        logging.info(f"Esquema correcto. Las {len(expected_columns)} columnas obligatorias estan presentes.")
        return {"valid": True, "extra_columns": list(extra)}

    # =========================================================================
    # 4. VALIDAR CALIDAD DE DATOS
    # =========================================================================

    @task
    def validar_calidad_datos(batch_id: int, data: dict):
        if data is None or batch_id is None:
            logging.info("No hay datos que validar porque el batch vino vacio o no se almaceno.")
            return {"valid": False, "reason": "sin datos"}

        records = data["data"]
        total   = len(records)
        logging.info(f"Validando calidad de {total} registros del batch_id={batch_id}...")

        invalid_price     = sum(1 for r in records if not r.get("price") or r["price"] <= 0)
        price_invalid_pct = invalid_price / total

        nulls = {
            col: sum(1 for r in records if r.get(col) is None)
            for col in ["bed", "bath", "acre_lot", "house_size"]
        }

        logging.info(f"Precios invalidos (nulos o <= 0): {invalid_price} de {total} ({price_invalid_pct:.1%})")
        logging.info(f"Nulos por columna numerica: {nulls}")

        if price_invalid_pct > 0.20:
            logging.error(
                f"Calidad insuficiente: {price_invalid_pct:.1%} de los registros no tienen precio valido. "
                f"El umbral maximo es 20%. El batch no sera usado para entrenamiento."
            )
            return {"valid": False, "reason": f"demasiados precios invalidos: {price_invalid_pct:.1%}"}

        logging.info("Calidad aceptable. El batch puede avanzar al siguiente paso del pipeline.")
        return {
            "valid": True,
            "total_records": total,
            "invalid_price_count": invalid_price,
            "nulls_per_column": nulls,
        }

    # =========================================================================
    # 5. DETECTAR NUEVAS CATEGORIAS
    # Compara las categorias del batch actual contra el historico en la BD.
    # Solo cuenta categorias que aparecen >= MIN_CATEGORY_FREQ veces.
    # =========================================================================

    @task
    def detectar_nuevas_categorias(data: dict, batch_id: int):
        if data is None or batch_id is None:
            return {"found": False, "new_categories": {}}

        from collections import Counter
        CATEGORICAL_COLS = ["status", "city", "state", "zip_code"]

        # Frecuencia de cada valor en el batch actual
        batch_cats = {
            col: Counter(r.get(col) for r in data["data"] if r.get(col) is not None)
            for col in CATEGORICAL_COLS
        }

        # Valores historicos vistos en batches anteriores
        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()

        historical_cats = {}
        for col in CATEGORICAL_COLS:
            cursor.execute(
                f"SELECT DISTINCT {col} FROM raw.properties WHERE batch_id < %s AND {col} IS NOT NULL",
                (batch_id,),
            )
            historical_cats[col] = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Nuevas categorias con frecuencia suficiente para afectar el modelo
        new_cats = {}
        for col in CATEGORICAL_COLS:
            nuevas = {
                val: cnt
                for val, cnt in batch_cats[col].items()
                if val not in historical_cats[col] and cnt >= MIN_CATEGORY_FREQ
            }
            if nuevas:
                new_cats[col] = nuevas

        if new_cats:
            total_new = sum(len(v) for v in new_cats.values())
            logging.warning(f"Se encontraron {total_new} categorias nuevas relevantes:")
            for col, vals in new_cats.items():
                logging.warning(f"  {col}: {list(vals.keys())}")
        else:
            logging.info("No se detectaron categorias nuevas relevantes en este batch.")

        return {"found": bool(new_cats), "new_categories": new_cats}

    # =========================================================================
    # 6. DETECTAR DRIFT
    # KS test entre la distribucion del batch actual y el historico acumulado.
    # Se evaluan las columnas numericas clave: price, bed, bath, acre_lot, house_size.
    # =========================================================================

    @task
    def detectar_drift(data: dict, batch_id: int):
        if data is None or batch_id is None:
            return {"detected": False, "columns": {}, "reason": "sin datos"}

        from scipy import stats

        NUMERIC_COLS = ["price", "bed", "bath", "acre_lot", "house_size"]

        batch_vals = {
            col: [r[col] for r in data["data"] if r.get(col) is not None and r[col] > 0]
            for col in NUMERIC_COLS
        }

        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()

        hist_vals = {}
        for col in NUMERIC_COLS:
            cursor.execute(
                f"SELECT {col} FROM raw.properties WHERE batch_id < %s AND {col} IS NOT NULL AND {col} > 0",
                (batch_id,),
            )
            hist_vals[col] = [row[0] for row in cursor.fetchall()]
        conn.close()

        if all(len(v) == 0 for v in hist_vals.values()):
            logging.info("Sin datos historicos todavia. No se puede calcular drift para el primer batch.")
            return {"detected": False, "columns": {}, "reason": "primer batch, sin historico"}

        drift_results = {}
        drift_detected = False

        for col in NUMERIC_COLS:
            curr = batch_vals.get(col, [])
            hist = hist_vals.get(col, [])

            if len(curr) < 30 or len(hist) < 30:
                logging.info(f"{col}: muestras insuficientes para KS test (actual={len(curr)}, hist={len(hist)})")
                drift_results[col] = {"drift": False, "reason": "muestras insuficientes"}
                continue

            ks_stat, p_value = stats.ks_2samp(curr, hist)
            has_drift = p_value < DRIFT_ALPHA

            if has_drift:
                drift_detected = True
                logging.warning(f"{col}: DRIFT detectado. KS={ks_stat:.4f}, p={p_value:.6f} (umbral={DRIFT_ALPHA})")
            else:
                logging.info(f"{col}: sin drift. KS={ks_stat:.4f}, p={p_value:.6f}")

            drift_results[col] = {"drift": has_drift, "ks_stat": float(ks_stat), "p_value": float(p_value)}

        if drift_detected:
            cols_con_drift = [c for c, r in drift_results.items() if r.get("drift")]
            logging.warning(f"Drift detectado en {len(cols_con_drift)} columna(s): {cols_con_drift}")
        else:
            logging.info("No se detecto drift estadisticamente significativo en este batch.")

        return {"detected": drift_detected, "columns": drift_results}

    # =========================================================================
    # 7. PREPROCESAR DATOS
    # Limpia el lote y lo escribe en clean.properties.
    # Solo corre si el esquema y la calidad son validos.
    # =========================================================================

    @task
    def preprocesar_datos(data: dict, batch_id: int, schema_ok: dict, quality_ok: dict):
        # Guardamos los datos CRUDOS tal como llegan en raw.properties (tarea anterior).
        # Esta tarea genera la version LIMPIA para entrenamiento en clean.properties.
        # Solo corre si el esquema y la calidad pasaron las validaciones anteriores.
        if data is None or batch_id is None:
            logging.info("Sin datos que preprocesar.")
            return {"valid": False, "records_processed": 0}

        if not schema_ok.get("valid") or not quality_ok.get("valid"):
            logging.warning(
                f"Validaciones fallidas (schema={schema_ok.get('valid')}, "
                f"quality={quality_ok.get('valid')}). No se preprocesan los datos."
            )
            return {"valid": False, "records_processed": 0}

        import pandas as pd

        logging.info(f"Iniciando preprocesamiento de {len(data['data'])} registros del batch_id={batch_id}.")
        logging.info("Objetivo: limpiar el lote crudo de raw.properties y escribir la version lista para entrenamiento en clean.properties.")

        df = pd.DataFrame(data["data"])
        logging.info(f"DataFrame creado con {len(df)} filas y {len(df.columns)} columnas.")

        # --- PASO 1: filtrar registros sin precio valido ---
        # price es el target del modelo (lo que queremos predecir).
        # Un registro sin precio no aporta al entrenamiento y lo descartamos.
        logging.info("PASO 1/4 — Eliminando registros con precio nulo o invalido (<= 0)...")
        before = len(df)
        df = df[df["price"].notna() & (df["price"] > 0)]
        removed = before - len(df)
        if removed > 0:
            logging.info(f"  Se eliminaron {removed} registros sin precio valido. Quedan {len(df)}.")
        else:
            logging.info(f"  Todos los registros tienen precio valido. Sin eliminaciones.")

        # --- PASO 2: imputar nulos en columnas numericas con la mediana ---
        # No descartamos registros con bed/bath/etc nulos porque perderiamos
        # demasiados datos. En cambio los rellenamos con la mediana del batch.
        # Usamos mediana (no promedio) porque es robusta ante valores extremos.
        logging.info("PASO 2/4 — Imputando nulos en columnas numericas (bed, bath, acre_lot, house_size) con la mediana del batch...")
        for col in ["bed", "bath", "acre_lot", "house_size"]:
            n_null = df[col].isna().sum()
            if n_null > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logging.info(f"  {col}: {n_null} nulos → mediana del batch = {median_val:.2f}")
            else:
                logging.info(f"  {col}: sin nulos.")

        # --- PASO 3: imputar nulos en columnas categoricas con 'unknown' ---
        # Para texto no existe mediana. Usamos 'unknown' como categoria especial.
        # El modelo esta entrenado para manejar esta categoria sin fallar.
        logging.info("PASO 3/4 — Imputando nulos en columnas categoricas (status, city, state, zip_code) con 'unknown'...")
        for col in ["status", "city", "state", "zip_code"]:
            n_null = df[col].isna().sum()
            if n_null > 0:
                df[col] = df[col].fillna("unknown")
                logging.info(f"  {col}: {n_null} nulos → 'unknown'")
            else:
                logging.info(f"  {col}: sin nulos.")

        # --- PASO 4: escribir en clean.properties ---
        # Solo guardamos las columnas que usa el modelo (no todas las del raw).
        # brokered_by y street no se incluyen porque son codigos sin valor predictivo claro.
        logging.info("PASO 4/4 — Escribiendo registros limpios en clean.properties...")
        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()

        records = [
            (
                batch_id,
                float(row["price"]),
                float(row["bed"])        if pd.notna(row["bed"])        else None,
                float(row["bath"])       if pd.notna(row["bath"])       else None,
                float(row["acre_lot"])   if pd.notna(row["acre_lot"])   else None,
                float(row["house_size"]) if pd.notna(row["house_size"]) else None,
                str(row["status"]),
                str(row["city"]),
                str(row["state"]),
                str(row["zip_code"]),
                str(row["prev_sold_date"]) if pd.notna(row.get("prev_sold_date")) else None,
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(
            """
            INSERT INTO clean.properties
                (batch_id, price, bed, bath, acre_lot, house_size,
                 status, city, state, zip_code, prev_sold_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )

        conn.commit()
        conn.close()
        logging.info(f"{len(records)} registros del batch {batch_id} escritos en clean.properties.")
        return {"valid": True, "records_processed": len(records)}

    # =========================================================================
    # 8. DECIDIR ENTRENAMIENTO (bifurcacion)
    # Reglas tecnicas basadas en drift, categorias nuevas, volumen y calidad.
    # =========================================================================

    @task.branch
    def decidir_entrenamiento(
        batch_id: int,
        schema_ok: dict,
        quality_ok: dict,
        nuevas_cat: dict,
        drift: dict,
        preproceso: dict,
    ):
        def _elegir(rama: str, motivo: str):
            otra = "entrenar_modelo" if rama == "omitir_entrenamiento" else "omitir_entrenamiento"
            logging.info("=" * 60)
            logging.info(f"DECISION DE ENTRENAMIENTO")
            logging.info(f"  Motivo  : {motivo}")
            logging.info(f"  Elegida : {rama}")
            logging.info(f"  Saltada : {otra}  <-- aparecera como [skipped] en la UI, eso es correcto")
            logging.info("=" * 60)
            return rama

        if batch_id is None:
            return _elegir("omitir_entrenamiento", "No hay batch_id valido.")

        if not schema_ok.get("valid"):
            return _elegir("omitir_entrenamiento", "Esquema invalido: faltan columnas obligatorias.")

        if not quality_ok.get("valid"):
            return _elegir("omitir_entrenamiento", "Calidad de datos insuficiente.")

        if not preproceso.get("valid"):
            return _elegir("omitir_entrenamiento", "El preprocesamiento fallo o no produjo registros.")

        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw.batch_audit")
        audit_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM clean.properties")
        total_clean = cursor.fetchone()[0]
        conn.close()

        # Regla 1: primer batch — entrenar para establecer linea base
        if audit_count == 0:
            return _elegir(
                "entrenar_modelo",
                f"Primer batch del pipeline. {total_clean} registros limpios. Se establece la linea base."
            )

        # Regla 2: volumen insuficiente — no entrenar
        if total_clean < MIN_RECORDS_TO_TRAIN:
            return _elegir(
                "omitir_entrenamiento",
                f"Solo hay {total_clean} registros limpios acumulados (minimo={MIN_RECORDS_TO_TRAIN})."
            )

        # Regla 3: drift detectado en alguna columna — entrenar
        if drift.get("detected"):
            cols_drift = [c for c, r in drift.get("columns", {}).items() if r.get("drift")]
            return _elegir("entrenar_modelo", f"Drift estadistico detectado en columnas: {cols_drift}.")

        # Regla 4: categorias nuevas relevantes — entrenar
        if nuevas_cat.get("found"):
            return _elegir("entrenar_modelo", "Se detectaron categorias nuevas relevantes en el batch.")

        return _elegir(
            "omitir_entrenamiento",
            f"Sin drift ni categorias nuevas. {total_clean} registros limpios acumulados. No justifica reentrenar."
        )

    # =========================================================================
    # 9. OMITIR ENTRENAMIENTO
    # Rama cuando se decide no entrenar. Registra la decision en batch_audit.
    # =========================================================================

    @task
    def omitir_entrenamiento(batch_id: int):
        if batch_id is None:
            return

        logging.info(f"Entrenamiento omitido para batch_id={batch_id}. Registrando en batch_audit...")

        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO raw.batch_audit (batch_id, decision, decision_reason)
            VALUES (%s, %s, %s)
            """,
            (batch_id, "skip", "Sin cambios significativos o validaciones fallidas"),
        )
        conn.commit()
        conn.close()
        logging.info("Decision 'skip' registrada en batch_audit.")

    # =========================================================================
    # 10. ENTRENAR MODELO
    # Entrena un RandomForestRegressor sobre todos los datos limpios acumulados.
    # Registra el modelo y parametros en MLflow.
    # =========================================================================

    @task
    def entrenar_modelo(batch_id: int):
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        NUMERIC_FEATS = ["bed", "bath", "acre_lot", "house_size"]
        CAT_FEATS     = ["status", "city", "state", "zip_code"]

        logging.info(f"Cargando todos los datos limpios acumulados para entrenar (batch actual={batch_id})...")

        hook   = PostgresHook(postgres_conn_id="postgres_mlops")
        engine = hook.get_sqlalchemy_engine()
        df     = pd.read_sql(
            "SELECT price, bed, bath, acre_lot, house_size, status, city, state, zip_code FROM clean.properties",
            engine,
        )

        logging.info(f"Total registros disponibles para entrenamiento: {len(df)}")

        X = df[NUMERIC_FEATS + CAT_FEATS]
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Train: {len(X_train)} registros | Test: {len(X_test)} registros")

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUMERIC_FEATS),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), CAT_FEATS),
        ])

        # n_estimators=50 y max_depth=15 reducen el uso de memoria significativamente
        # frente a los defaults (100 arboles sin limite de profundidad) sin sacrificar
        # mucho desempeno en un dataset de este tamano.
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor",    RandomForestRegressor(
                n_estimators=50,
                max_depth=15,
                random_state=42,
                n_jobs=1,
            )),
        ])

        logging.info("Entrenando RandomForestRegressor (n_estimators=100)...")
        model.fit(X_train, y_train)

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("batch_id",      batch_id)
            mlflow.log_param("n_estimators",  100)
            mlflow.log_param("train_records", len(X_train))
            mlflow.log_param("test_records",  len(X_test))
            mlflow.log_param("features",      str(NUMERIC_FEATS + CAT_FEATS))
            mlflow.sklearn.log_model(model, artifact_path="model")

        logging.info(f"Modelo entrenado y guardado en MLflow. run_id={run_id}")
        return run_id

    # =========================================================================
    # 11. EVALUAR MODELO
    # Calcula MAE, RMSE y R2 sobre el conjunto de test y los registra en MLflow.
    # =========================================================================

    @task
    def evaluar_modelo(batch_id: int, mlflow_run_id: str):
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        logging.info(f"Evaluando modelo mlflow_run_id={mlflow_run_id}...")

        NUMERIC_FEATS = ["bed", "bath", "acre_lot", "house_size"]
        CAT_FEATS     = ["status", "city", "state", "zip_code"]

        hook   = PostgresHook(postgres_conn_id="postgres_mlops")
        engine = hook.get_sqlalchemy_engine()
        df     = pd.read_sql(
            "SELECT price, bed, bath, acre_lot, house_size, status, city, state, zip_code FROM clean.properties",
            engine,
        )

        X = df[NUMERIC_FEATS + CAT_FEATS]
        y = df["price"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.set_tracking_uri(MLFLOW_URI)
        model  = mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/model")
        y_pred = model.predict(X_test)

        mae  = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))

        logging.info(f"Metricas del candidato — MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2",   r2)

        return {"run_id": mlflow_run_id, "mae": mae, "rmse": rmse, "r2": r2}

    # =========================================================================
    # 12. COMPARAR CON EL MODELO EN PRODUCCION
    # Si no existe modelo productivo, el candidato se promueve automaticamente.
    # Si existe, compara MAE y RMSE.
    # =========================================================================

    @task
    def comparar_con_produccion(batch_id: int, candidate: dict):
        import mlflow
        from mlflow import MlflowClient

        logging.info(f"Comparando candidato (run_id={candidate['run_id']}) contra el modelo productivo...")

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        try:
            prod_version = client.get_model_version_by_alias(EXPERIMENT_NAME, PROD_ALIAS)
            prod_run     = client.get_run(prod_version.run_id)
            prod_mae     = float(prod_run.data.metrics.get("mae",  float("inf")))
            prod_rmse    = float(prod_run.data.metrics.get("rmse", float("inf")))
            logging.info(f"Modelo productivo — MAE: {prod_mae:.2f} | RMSE: {prod_rmse:.2f}")
        except Exception:
            logging.info("No existe modelo productivo todavia. El candidato se promovera automaticamente.")
            return {
                "has_production": False,
                "should_promote": True,
                "reason": "primer modelo, no existe produccion previa",
                "candidate": candidate,
            }

        mae_mejora  = (prod_mae  - candidate["mae"])  / prod_mae
        rmse_cambio = (candidate["rmse"] - prod_rmse) / prod_rmse

        logging.info(f"Mejora en MAE:   {mae_mejora:+.2%} (umbral minimo: +{MAE_IMPROVE_PCT:.0%})")
        logging.info(f"Cambio en RMSE:  {rmse_cambio:+.2%} (maximo tolerable: +{MAX_RMSE_DEGRADE_PCT:.0%})")

        should_promote = (mae_mejora >= MAE_IMPROVE_PCT) and (rmse_cambio <= MAX_RMSE_DEGRADE_PCT)

        if should_promote:
            reason = (
                f"Candidato mejora MAE en {mae_mejora:.2%} "
                f"y RMSE solo varia {rmse_cambio:.2%}"
            )
            logging.info(f"El candidato SUPERA al productivo. {reason}")
        else:
            reason = (
                f"Candidato no supera al productivo — "
                f"MAE mejoro {mae_mejora:.2%} (necesita ≥{MAE_IMPROVE_PCT:.0%}) | "
                f"RMSE cambio {rmse_cambio:.2%}"
            )
            logging.info(f"El candidato NO supera al productivo. {reason}")

        return {
            "has_production": True,
            "should_promote": should_promote,
            "reason": reason,
            "candidate": candidate,
            "production_mae": prod_mae,
            "production_rmse": prod_rmse,
        }

    # =========================================================================
    # 13. DECIDIR PROMOCION (bifurcacion)
    # =========================================================================

    @task.branch
    def decidir_promocion(batch_id: int, comparison: dict):
        if comparison.get("should_promote"):
            otra = "rechazar_modelo"
            elegida = "promover_modelo"
            motivo = comparison.get("reason", "el candidato supera al modelo productivo")
        else:
            otra = "promover_modelo"
            elegida = "rechazar_modelo"
            motivo = comparison.get("reason", "el candidato no supera al modelo productivo")

        logging.info("=" * 60)
        logging.info(f"DECISION DE PROMOCION")
        logging.info(f"  Motivo  : {motivo}")
        logging.info(f"  Elegida : {elegida}")
        logging.info(f"  Saltada : {otra}  <-- aparecera como [skipped] en la UI, eso es correcto")
        logging.info("=" * 60)
        return elegida

    # =========================================================================
    # 14. PROMOVER MODELO
    # Registra el modelo en MLflow y le asigna el alias de produccion.
    # =========================================================================

    @task
    def promover_modelo(batch_id: int, comparison: dict):
        import mlflow
        from mlflow import MlflowClient

        candidate = comparison["candidate"]
        run_id    = candidate["run_id"]
        reason    = comparison["reason"]

        logging.info(f"Promoviendo modelo run_id={run_id} como '{PROD_ALIAS}'...")

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        try:
            client.create_registered_model(EXPERIMENT_NAME)
        except Exception:
            pass

        mv = mlflow.register_model(f"runs:/{run_id}/model", EXPERIMENT_NAME)
        client.set_registered_model_alias(EXPERIMENT_NAME, PROD_ALIAS, mv.version)

        logging.info(f"Modelo v{mv.version} registrado. Alias '{PROD_ALIAS}' actualizado.")

        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO raw.batch_audit
                (batch_id, decision, decision_reason,
                 candidate_run_id, candidate_mae, candidate_rmse, candidate_r2,
                 promoted, promotion_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                batch_id, "train", reason,
                run_id, candidate["mae"], candidate["rmse"], candidate["r2"],
                True, reason,
            ),
        )
        conn.commit()
        conn.close()

    # =========================================================================
    # 15. RECHAZAR MODELO
    # El candidato no supera al productivo. Queda registrado en MLflow pero
    # no se actualiza el alias de produccion.
    # =========================================================================

    @task
    def rechazar_modelo(batch_id: int, comparison: dict):
        candidate = comparison["candidate"]
        run_id    = candidate["run_id"]
        reason    = comparison["reason"]

        logging.warning(f"Modelo run_id={run_id} rechazado. Razon: {reason}")

        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO raw.batch_audit
                (batch_id, decision, decision_reason,
                 candidate_run_id, candidate_mae, candidate_rmse, candidate_r2,
                 promoted, promotion_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                batch_id, "train", f"Entrenado pero rechazado",
                run_id, candidate["mae"], candidate["rmse"], candidate["r2"],
                False, reason,
            ),
        )
        conn.commit()
        conn.close()
        logging.info("Rechazo registrado en batch_audit. El modelo productivo actual se mantiene.")

    # =========================================================================
    # 16. REGISTRAR RESULTADO FINAL
    # Lee el estado final de batch_audit y lo resume en los logs.
    # Usa trigger_rule NONE_FAILED_MIN_ONE_SUCCESS para correr despues de
    # cualquiera de las ramas (skip, promover, rechazar).
    # =========================================================================

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def registrar_resultado(batch_id: int):
        if batch_id is None:
            logging.info("Sin batch_id. No hay resultado que registrar.")
            return

        hook = PostgresHook(postgres_conn_id="postgres_mlops")
        conn = hook.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT decision, decision_reason, candidate_run_id,
                   candidate_mae, candidate_rmse, candidate_r2,
                   promoted, promotion_reason
            FROM raw.batch_audit
            WHERE batch_id = %s
            ORDER BY executed_at DESC
            LIMIT 1
            """,
            (batch_id,),
        )
        row = cursor.fetchone()
        conn.close()

        logging.info("=" * 60)
        logging.info(f"RESULTADO FINAL — batch_id={batch_id}")
        if row:
            decision, reason, run_id, mae, rmse, r2, promoted, prom_reason = row
            logging.info(f"  Decision:    {decision}")
            logging.info(f"  Razon:       {reason}")
            if run_id:
                logging.info(f"  Candidato:   {run_id}")
                logging.info(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")
                logging.info(f"  Promovido:   {promoted}")
                if prom_reason:
                    logging.info(f"  Detalle:     {prom_reason}")
        else:
            logging.info("  (sin registro en batch_audit)")
        logging.info("=" * 60)

    # =========================================================================
    # DEFINICION DEL FLUJO
    # =========================================================================

    batch_data = obtener_lote_api()
    batch_id   = guardar_datos_postgres(batch_data)

    # Validaciones y detecciones corren en paralelo
    schema_ok  = validar_columnas(batch_data)
    quality_ok = validar_calidad_datos(batch_id, batch_data)
    nuevas_cat = detectar_nuevas_categorias(batch_data, batch_id)
    drift      = detectar_drift(batch_data, batch_id)
    preproceso = preprocesar_datos(batch_data, batch_id, schema_ok, quality_ok)

    # Bifurcacion: entrenar o no
    branch1    = decidir_entrenamiento(batch_id, schema_ok, quality_ok, nuevas_cat, drift, preproceso)
    skip       = omitir_entrenamiento(batch_id)
    mlflow_run_id = entrenar_modelo(batch_id)
    metrics       = evaluar_modelo(batch_id, mlflow_run_id)
    comparison    = comparar_con_produccion(batch_id, metrics)

    # Bifurcacion: promover o rechazar
    branch2  = decidir_promocion(batch_id, comparison)
    promote  = promover_modelo(batch_id, comparison)
    reject   = rechazar_modelo(batch_id, comparison)

    # Convergencia final
    fin = registrar_resultado(batch_id)

    # Control flow de las bifurcaciones
    branch1 >> [skip, mlflow_run_id]
    branch2 >> [promote, reject]
    [skip, promote, reject] >> fin

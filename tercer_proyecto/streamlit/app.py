import os
import requests
import psycopg2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://inference-service.inference.svc.cluster.local:8000")
DB_HOST     = os.environ.get("DB_HOST", "postgres-service.postgres.svc.cluster.local")
DB_PORT     = os.environ.get("DB_PORT", "5432")
DB_NAME     = os.environ.get("DB_NAME", "mlops")
DB_USER     = os.environ.get("DB_USER", "grupo4")
DB_PASS     = os.environ.get("DB_PASS", "grupo4pass")

st.set_page_config(page_title="MLOps Dashboard — Grupo 4", layout="wide")


@st.cache_data(ttl=300)
def cargar_ubicaciones():
    """Trae todas las combinaciones únicas de estado, ciudad y código postal
    que existen en los datos limpios. Se cachea 5 minutos para no consultar
    la BD en cada interacción del usuario."""
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    df = pd.read_sql(
        "SELECT DISTINCT state, city, zip_code FROM clean.properties WHERE state IS NOT NULL AND city IS NOT NULL AND zip_code IS NOT NULL ORDER BY state, city, zip_code",
        conn,
    )
    conn.close()
    # Los zip codes vienen como "75402.0" porque pandas los leyó como float. Los limpiamos.
    df["zip_code"] = df["zip_code"].str.replace(r"\.0$", "", regex=True)
    return df
st.title("Dashboard MLOps — Precio de Propiedades")
st.markdown("Pontificia Universidad Javeriana · Grupo 4 · Thomas Rivera & Laura Sotto")

# ── Barra lateral con estado de la API y el modelo ────────────────────────
with st.sidebar:
    st.header("Estado del servicio")

    # Chequeamos /health cada vez que la app carga para saber si la API responde
    try:
        h = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        if h.status_code == 200 and h.json().get("model_loaded"):
            st.success("API en línea con modelo cargado")
        elif h.status_code == 200:
            st.warning("API en línea pero sin modelo")
        else:
            st.error("API con problemas")
    except Exception:
        st.error("API no disponible")

    st.divider()
    st.header("Modelo en producción")

    # El botón consulta /model-info para ver qué versión está sirviendo ahora mismo
    if st.button("Actualizar info del modelo"):
        try:
            r = requests.get(f"{FASTAPI_URL}/model-info", timeout=5)
            if r.status_code == 200:
                info = r.json()
                st.write(f"**Nombre:** {info['model_name']}")
                st.write(f"**Versión:** {info['version']}")
                st.write(f"**Alias:** {info['alias']}")
                st.write(f"**Run ID:** `{info['run_id'][:8]}...`")
            elif r.status_code == 503:
                st.warning("No hay modelo en producción todavía.")
            else:
                st.error(f"Error {r.status_code}")
        except Exception as e:
            st.error(f"No se puede conectar: {e}")

    st.divider()
    st.image("Casa.jpg", use_container_width=True)
    st.caption(f"API: `{FASTAPI_URL}`")

tab_prediccion, tab_pipeline = st.tabs(["Predecir precio", "Historial del pipeline"])


# ──────────────────────────────────────────────────────────────
# PESTAÑA 1 — Formulario de predicción
# ──────────────────────────────────────────────────────────────
with tab_prediccion:
    st.subheader("Estimar el precio de una propiedad")

    # Propiedad de ejemplo tomada directamente del batch 1 del dataset real
    EJEMPLO = {
        "bed": 3, "bath": 2, "acre_lot": 0.25, "house_size": 1800,
        "status": "for_sale", "city": "Austin", "state": "Texas", "zip_code": "78701",
    }

    # Inicializamos el session_state con los valores del ejemplo.
    # Esto permite que el botón "Cargar ejemplo" llene el formulario sin que
    # Streamlit ignore los cambios al hacer rerun.
    for k, v in EJEMPLO.items():
        if f"w_{k}" not in st.session_state:
            st.session_state[f"w_{k}"] = v

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        if st.button("Cargar datos de ejemplo"):
            for k, v in EJEMPLO.items():
                st.session_state[f"w_{k}"] = v
            # Los selects de ciudad y zip usan claves propias porque no tienen key= de widget
            st.session_state["w_city_val"] = EJEMPLO["city"]
            st.session_state["w_zip_val"]  = EJEMPLO["zip_code"]
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Características físicas**")
        bed        = st.number_input("Habitaciones",        min_value=0, max_value=20, key="w_bed")
        bath       = st.number_input("Baños",               min_value=0, max_value=20, key="w_bath")
        acre_lot   = st.number_input("Terreno (acres)",     min_value=0.0, step=0.05, format="%.2f", key="w_acre_lot")
        house_size = st.number_input("Área habitable (ft²)", min_value=0, step=50, key="w_house_size")

    with col2:
        st.markdown("**Ubicación y estado**")
        status = st.selectbox("Estado de la propiedad", ["for_sale", "sold"], key="w_status")

        # Cargamos todas las combinaciones reales del dataset
        df_geo = cargar_ubicaciones()
        estados = sorted(df_geo["state"].unique())

        # Selectbox de estado. key="w_state" permite que el botón "Cargar ejemplo"
        # lo llene vía session_state.
        state_sel = st.selectbox("Estado (USA)", estados, key="w_state",
                                 index=estados.index(st.session_state["w_state"]) if st.session_state.get("w_state") in estados else 0)

        # Filtramos ciudades al estado seleccionado
        ciudades = sorted(df_geo[df_geo["state"] == state_sel]["city"].unique())
        # Si la ciudad guardada en session_state no pertenece al nuevo estado, usamos la primera
        ciudad_actual = st.session_state.get("w_city_val", ciudades[0])
        ciudad_idx = ciudades.index(ciudad_actual) if ciudad_actual in ciudades else 0
        city_sel = st.selectbox("Ciudad", ciudades, index=ciudad_idx)
        st.session_state["w_city_val"] = city_sel

        # Filtramos zip codes al estado y ciudad seleccionados
        zips = sorted(df_geo[(df_geo["state"] == state_sel) & (df_geo["city"] == city_sel)]["zip_code"].unique())
        zip_actual = st.session_state.get("w_zip_val", zips[0])
        zip_idx = zips.index(zip_actual) if zip_actual in zips else 0
        zip_sel = st.selectbox("Código postal", zips, index=zip_idx)
        st.session_state["w_zip_val"] = zip_sel

        state    = state_sel
        city     = city_sel
        zip_code = zip_sel

    st.divider()
    if st.button("Estimar precio", type="primary", use_container_width=True):
        payload = {
            "bed": bed, "bath": bath, "acre_lot": acre_lot, "house_size": house_size,
            "status": status, "city": city, "state": state, "zip_code": zip_code,
        }
        with st.spinner("Consultando API..."):
            try:
                resp = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=30)

                if resp.status_code == 200:
                    data  = resp.json()
                    price = data["predicted_price"]
                    st.success(f"Precio estimado: **${price:,.0f} USD**")

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric("Precio estimado", f"${price:,.0f}")
                    with col_r2:
                        st.metric("Versión del modelo", data["model_version"])

                    # Detalles del modelo en un expander para no saturar la pantalla
                    with st.expander("Detalles del modelo"):
                        st.write(f"**Nombre:** {data['model_name']}")
                        st.write(f"**Versión:** {data['model_version']}")

                elif resp.status_code == 503:
                    st.error("No hay modelo en producción. Ejecuta primero el DAG de Airflow.")
                elif resp.status_code == 422:
                    st.error("Error de validación: revisa los valores ingresados.")
                    st.json(resp.json())
                else:
                    st.error(f"La API respondió con estado {resp.status_code}")
                    st.write(resp.text)

            except requests.exceptions.ConnectionError:
                st.error(f"No se puede conectar a la API en {FASTAPI_URL}.")
            except requests.exceptions.Timeout:
                st.error("La solicitud tardó demasiado (timeout).")
            except Exception as e:
                st.error(f"Error inesperado: {e}")


# ──────────────────────────────────────────────────────────────
# PESTAÑA 2 — Historial del pipeline
# ──────────────────────────────────────────────────────────────
with tab_pipeline:
    st.subheader("Historial de batches procesados por Airflow")

    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        df = pd.read_sql("""
            SELECT
                b.batch_number   AS batch,
                a.executed_at    AS fecha,
                a.decision,
                a.promoted       AS promovido,
                a.candidate_mae  AS mae,
                a.candidate_rmse AS rmse,
                a.candidate_r2   AS r2,
                a.decision_reason AS motivo
            FROM raw.batch_audit a
            JOIN raw.batches b ON b.id = a.batch_id
            ORDER BY a.id
        """, conn)
        df_inf = pd.read_sql("""
            SELECT COUNT(*) AS total, AVG(prediction) AS precio_promedio
            FROM raw.inference_events
            WHERE response_status = 'ok'
        """, conn)
        conn.close()
    except Exception as exc:
        st.error(f"No se pudo conectar a la base de datos: {exc}")
        st.stop()

    # ── Métricas resumen ────────────────────────────────────
    # st.columns(4) divide la pantalla en 4 columnas iguales lado a lado.
    # Cada variable (m1, m2...) es una columna donde podemos escribir cosas.
    m1, m2, m3, m4 = st.columns(4)

    # len(df) cuenta cuántas filas tiene el dataframe, una por cada batch en batch_audit.
    m1.metric("Batches procesados", len(df))

    # df["decision"] == "train" crea una serie de True/False por cada fila.
    # .sum() suma esos True (cada True vale 1) para contar cuántos fueron "train".
    m2.metric("Modelos entrenados", int((df["decision"] == "train").sum()))

    # Mismo truco: suma los True de la columna promovido para saber cuántos modelos llegaron a producción.
    m3.metric("Modelos promovidos", int(df["promovido"].sum()))

    # df_inf es el resultado del SELECT COUNT(*) de inference_events, .iloc[0] toma el primer (y único) valor.
    m4.metric("Predicciones totales", int(df_inf["total"].iloc[0]))

    st.divider()

    # ── Tabla de historial ───────────────────────────────────
    st.write("**Detalle por batch**")
    tabla = df.copy()
    tabla["fecha"]     = pd.to_datetime(tabla["fecha"]).dt.strftime("%Y-%m-%d %H:%M")
    tabla["mae"]       = tabla["mae"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    tabla["rmse"]      = tabla["rmse"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    tabla["r2"]        = tabla["r2"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    tabla["promovido"] = tabla["promovido"].map({True: "sí", False: "no"}).fillna("—")

    st.dataframe(
        tabla.rename(columns={
            "batch": "Batch", "fecha": "Fecha", "decision": "Decisión",
            "promovido": "Promovido", "mae": "MAE", "rmse": "RMSE",
            "r2": "R²", "motivo": "Motivo",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Gráfica de MAE por batch entrenado ──────────────────
    df_modelos = df[df["decision"] == "train"].dropna(subset=["mae"])
    if not df_modelos.empty:
        st.divider()
        st.write("**MAE por batch entrenado**")
        colores = ["#28a745" if p else "#dc3545" for p in df_modelos["promovido"].map({"sí": True, "no": False}).fillna(False)]
        fig = go.Figure(go.Bar(
            x=df_modelos["batch"].astype(str),
            y=df_modelos["mae"],
            marker_color=colores,
            text=df_modelos["mae"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Batch", yaxis_title="MAE (USD)",
            showlegend=False, height=350, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Verde = modelo promovido a producción · Rojo = modelo rechazado")

    # ── Últimas inferencias ──────────────────────────────────
    st.divider()
    st.write("**Últimas 10 predicciones recibidas**")
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        df_eventos = pd.read_sql("""
            SELECT requested_at, prediction, model_version, response_status
            FROM raw.inference_events
            ORDER BY requested_at DESC
            LIMIT 10
        """, conn)
        conn.close()
        df_eventos["requested_at"] = pd.to_datetime(df_eventos["requested_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_eventos["prediction"]   = df_eventos["prediction"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
        st.dataframe(
            df_eventos.rename(columns={
                "requested_at": "Fecha", "prediction": "Precio estimado",
                "model_version": "Versión", "response_status": "Estado",
            }),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"No se pudieron cargar las inferencias: {exc}")

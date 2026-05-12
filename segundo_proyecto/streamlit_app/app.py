import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:30800")

st.set_page_config(
    page_title="Predecir Reingreso por Diabetes",
    page_icon=":hospital:",
    layout="wide",
)

st.title("Predecir Reingreso por Diabetes")
st.markdown("MLOps Proyecto 2 — Pontificia Universidad Javeriana · Grupo 4")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("Estado del Modelo")
    if st.button("Actualizar info del modelo"):
        try:
            r = requests.get(f"{API_URL}/model-info", timeout=5)
            if r.status_code == 200:
                info = r.json()
                st.success("Modelo cargado")
                st.write(f"**Nombre:** {info['model_name']}")
                st.write(f"**Versión:** {info['model_version']}")
                st.write(f"**Alias:** {info['model_alias']}")
                st.write(f"**Run ID:** `{info['run_id'][:8]}...`")
            else:
                st.error(f"Error de API: {r.status_code}")
        except Exception as e:
            st.error(f"No se puede conectar a la API: {e}")

    st.divider()
    st.header("API")
    st.write(f"`{API_URL}`")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3)
        if h.status_code == 200:
            st.success("API en línea")
        else:
            st.warning("API con problemas")
    except Exception:
        st.error("API no disponible")

# ── Ejemplo ───────────────────────────────────
EXAMPLE = {
    "time_in_hospital": 3,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 13,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 9,
    "age_encoded": 65,
    "gender_encoded": 1,
    "race_encoded": 0,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "a1cresult_encoded": 0,
    "max_glu_serum_encoded": 0,
    "metformin_encoded": 1,
    "insulin_encoded": 1,
    "glipizide_encoded": 0,
    "glyburide_encoded": 0,
    "pioglitazone_encoded": 0,
    "rosiglitazone_encoded": 0,
    "change_encoded": 1,
    "diabetesmed_encoded": 1,
    "diag_1_code": 250.0,
    "diag_2_code": 401.0,
    "diag_3_code": 276.0,
}

# Inicializar session state con claves por widget
def init_state():
    for k, v in EXAMPLE.items():
        key = f"w_{k}"
        if key not in st.session_state:
            st.session_state[key] = v

init_state()

# Los widgets usan key="w_<campo>" para que al actualizar st.session_state y llamar
# st.rerun() los widgets tomen el nuevo valor. Sin key, Streamlit ignora el session_state
# y mantiene el valor que el usuario había ingresado.
col_load, _ = st.columns([1, 5])
with col_load:
    if st.button("Cargar datos de ejemplo"):
        for k, v in EXAMPLE.items():
            st.session_state[f"w_{k}"] = v
        st.rerun()

# ── Formulario ───────────────────────────────
st.subheader("Datos del Paciente")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Estadía hospitalaria**")
    time_in_hospital        = st.number_input("Días en hospital", 1, 14, key="w_time_in_hospital")
    num_lab_procedures      = st.number_input("Procedimientos de laboratorio", 0, 200, key="w_num_lab_procedures")
    num_procedures          = st.number_input("Procedimientos clínicos", 0, 10, key="w_num_procedures")
    num_medications         = st.number_input("Medicamentos", 0, 100, key="w_num_medications")
    number_diagnoses        = st.number_input("Diagnósticos", 1, 20, key="w_number_diagnoses")
    number_outpatient       = st.number_input("Visitas ambulatorias (año anterior)", 0, 50, key="w_number_outpatient")
    number_emergency        = st.number_input("Visitas a urgencias (año anterior)", 0, 50, key="w_number_emergency")
    number_inpatient        = st.number_input("Hospitalizaciones (año anterior)", 0, 50, key="w_number_inpatient")

with col2:
    st.markdown("**Demografía del paciente**")
    AGE_OPTIONS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    age_encoded = st.selectbox(
        "Grupo de edad (punto medio)",
        AGE_OPTIONS,
        index=AGE_OPTIONS.index(int(st.session_state["w_age_encoded"])),
        key="w_age_encoded",
    )
    gender_encoded = st.selectbox(
        "Género",
        [0, 1],
        format_func=lambda x: "Masculino" if x == 0 else "Femenino",
        key="w_gender_encoded",
    )
    race_encoded = st.selectbox(
        "Raza",
        [0, 1, 2, 3, 4],
        format_func=lambda x: ["Caucásico", "Afroamericano", "Hispano", "Asiático", "Otro"][x],
        key="w_race_encoded",
    )
    admission_type_id         = st.number_input("ID tipo de admisión (1–8)", 1, 8, key="w_admission_type_id")
    discharge_disposition_id  = st.number_input("ID disposición de alta (1–30)", 1, 30, key="w_discharge_disposition_id")
    admission_source_id       = st.number_input("ID fuente de admisión (1–26)", 1, 26, key="w_admission_source_id")

    st.markdown("**Resultados de laboratorio**")
    a1cresult_encoded = st.selectbox(
        "Resultado HbA1c",
        [0, 1, 2, 3],
        format_func=lambda x: ["Sin resultado", "Normal", ">7", ">8"][x],
        key="w_a1cresult_encoded",
    )
    max_glu_serum_encoded = st.selectbox(
        "Glucosa sérica máxima",
        [0, 1, 2, 3],
        format_func=lambda x: ["Sin resultado", "Normal", ">200", ">300"][x],
        key="w_max_glu_serum_encoded",
    )

with col3:
    st.markdown("**Medicamentos** (0=No, 1=Estable, 2=Aumentó, 3=Disminuyó)")
    med_map = lambda x: ["No", "Estable", "Aumentó", "Disminuyó"][int(x)]
    metformin_encoded    = st.selectbox("Metformina", [0, 1, 2, 3], format_func=med_map, key="w_metformin_encoded")
    insulin_encoded      = st.selectbox("Insulina", [0, 1, 2, 3], format_func=med_map, key="w_insulin_encoded")
    glipizide_encoded    = st.selectbox("Glipizida", [0, 1, 2, 3], format_func=med_map, key="w_glipizide_encoded")
    glyburide_encoded    = st.selectbox("Gliburida", [0, 1, 2, 3], format_func=med_map, key="w_glyburide_encoded")
    pioglitazone_encoded = st.selectbox("Pioglitazona", [0, 1, 2, 3], format_func=med_map, key="w_pioglitazone_encoded")
    rosiglitazone_encoded = st.selectbox("Rosiglitazona", [0, 1, 2, 3], format_func=med_map, key="w_rosiglitazone_encoded")

    st.markdown("**Otros**")
    change_encoded = st.selectbox(
        "Cambio en medicación",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
        key="w_change_encoded",
    )
    diabetesmed_encoded = st.selectbox(
        "Medicamento para diabetes recetado",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
        key="w_diabetesmed_encoded",
    )

    st.markdown("**Códigos de diagnóstico**")
    diag_1_code = st.number_input("Diagnóstico 1 (numérico)", 0.0, 999.0, key="w_diag_1_code")
    diag_2_code = st.number_input("Diagnóstico 2 (numérico)", 0.0, 999.0, key="w_diag_2_code")
    diag_3_code = st.number_input("Diagnóstico 3 (numérico)", 0.0, 999.0, key="w_diag_3_code")

# ── Predicción ───────────────────────────────
st.divider()
if st.button("Predecir Reingreso", type="primary", use_container_width=True):
    payload = {
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "age_encoded": age_encoded,
        "gender_encoded": gender_encoded,
        "race_encoded": race_encoded,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "a1cresult_encoded": a1cresult_encoded,
        "max_glu_serum_encoded": max_glu_serum_encoded,
        "metformin_encoded": metformin_encoded,
        "repaglinide_encoded": 0,
        "nateglinide_encoded": 0,
        "chlorpropamide_encoded": 0,
        "glimepiride_encoded": 0,
        "acetohexamide_encoded": 0,
        "glipizide_encoded": glipizide_encoded,
        "glyburide_encoded": glyburide_encoded,
        "tolbutamide_encoded": 0,
        "pioglitazone_encoded": pioglitazone_encoded,
        "rosiglitazone_encoded": rosiglitazone_encoded,
        "acarbose_encoded": 0,
        "miglitol_encoded": 0,
        "troglitazone_encoded": 0,
        "tolazamide_encoded": 0,
        "examide_encoded": 0,
        "citoglipton_encoded": 0,
        "insulin_encoded": insulin_encoded,
        "glyburide_metformin_encoded": 0,
        "glipizide_metformin_encoded": 0,
        "glimepiride_pioglitazone_encoded": 0,
        "metformin_rosiglitazone_encoded": 0,
        "metformin_pioglitazone_encoded": 0,
        "change_encoded": change_encoded,
        "diabetesmed_encoded": diabetesmed_encoded,
        "diag_1_code": diag_1_code,
        "diag_2_code": diag_2_code,
        "diag_3_code": diag_3_code,
    }

    with st.spinner("Consultando API..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if resp.status_code == 200:
                result = resp.json()
                st.divider()
                if result["prediction"] == 1:
                    st.error("RIESGO ALTO: El paciente probablemente será readmitido en menos de 30 días")
                else:
                    st.success("RIESGO BAJO: El paciente probablemente NO será readmitido en los próximos 30 días")

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Predicción", result["prediction_label"].replace("_", " "))
                with col_r2:
                    st.metric("Confianza", f"{result['probability']:.1%}")
                with col_r3:
                    st.metric("Tiempo de respuesta", f"{result['response_time_ms']:.1f} ms")

                with st.expander("Detalles del modelo"):
                    st.write(f"**Modelo:** {result['model_name']}")
                    st.write(f"**Versión:** {result['model_version']}")
                    st.write(f"**Alias:** {result['model_alias']}")
                    st.write(f"**ID de solicitud:** `{result['request_id']}`")

            elif resp.status_code == 422:
                st.error("Error de validación: revisa los valores ingresados.")
                st.json(resp.json())
            elif resp.status_code == 503:
                st.error("Modelo no disponible. Ejecuta primero el DAG de Airflow.")
            else:
                st.error(f"La API respondió con estado {resp.status_code}")
                st.write(resp.text)

        except requests.exceptions.ConnectionError:
            st.error(f"No se puede conectar a la API en {API_URL}. ¿Está el servicio corriendo?")
        except requests.exceptions.Timeout:
            st.error("La solicitud tardó demasiado (timeout).")
        except Exception as e:
            st.error(f"Error inesperado: {e}")

import streamlit as st
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://localhost:30800")

st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Diabetes Readmission Predictor")
st.markdown("MLOps Proyecto 2 — Pontificia Universidad Javeriana · Grupo 4")

# ── Sidebar: model info ──────────────────────
with st.sidebar:
    st.header("Model Status")
    if st.button("Refresh model info"):
        try:
            r = requests.get(f"{API_URL}/model-info", timeout=5)
            if r.status_code == 200:
                info = r.json()
                st.success("Model loaded")
                st.write(f"**Name:** {info['model_name']}")
                st.write(f"**Version:** {info['model_version']}")
                st.write(f"**Alias:** {info['model_alias']}")
                st.write(f"**Run ID:** `{info['run_id'][:8]}...`")
            else:
                st.error(f"API error: {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.divider()
    st.header("API")
    st.write(f"`{API_URL}`")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3)
        if h.status_code == 200:
            st.success("API healthy")
        else:
            st.warning("API unhealthy")
    except Exception:
        st.error("API unreachable")

# ── Example values ───────────────────────────
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
    "repaglinide_encoded": 0,
    "nateglinide_encoded": 0,
    "chlorpropamide_encoded": 0,
    "glimepiride_encoded": 0,
    "acetohexamide_encoded": 0,
    "glipizide_encoded": 0,
    "glyburide_encoded": 0,
    "tolbutamide_encoded": 0,
    "pioglitazone_encoded": 0,
    "rosiglitazone_encoded": 0,
    "acarbose_encoded": 0,
    "miglitol_encoded": 0,
    "troglitazone_encoded": 0,
    "tolazamide_encoded": 0,
    "examide_encoded": 0,
    "citoglipton_encoded": 0,
    "insulin_encoded": 1,
    "glyburide_metformin_encoded": 0,
    "glipizide_metformin_encoded": 0,
    "glimepiride_pioglitazone_encoded": 0,
    "metformin_rosiglitazone_encoded": 0,
    "metformin_pioglitazone_encoded": 0,
    "change_encoded": 1,
    "diabetesmed_encoded": 1,
    "diag_1_code": 250.0,
    "diag_2_code": 401.0,
    "diag_3_code": 276.0,
}

# ── Session state ─────────────────────────────
if "form_values" not in st.session_state:
    st.session_state["form_values"] = dict(EXAMPLE)

col_load, _ = st.columns([1, 5])
with col_load:
    if st.button("Load example values"):
        st.session_state["form_values"] = dict(EXAMPLE)
        st.rerun()

# ── Input form ───────────────────────────────
st.subheader("Patient Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Hospital stay**")
    v = st.session_state["form_values"]
    time_in_hospital = st.number_input("Time in hospital (days)", 1, 14, int(v["time_in_hospital"]))
    num_lab_procedures = st.number_input("Lab procedures", 0, 200, int(v["num_lab_procedures"]))
    num_procedures = st.number_input("Procedures", 0, 10, int(v["num_procedures"]))
    num_medications = st.number_input("Medications", 0, 100, int(v["num_medications"]))
    number_diagnoses = st.number_input("Diagnoses", 1, 20, int(v["number_diagnoses"]))
    number_outpatient = st.number_input("Outpatient visits (prev year)", 0, 50, int(v["number_outpatient"]))
    number_emergency = st.number_input("Emergency visits (prev year)", 0, 50, int(v["number_emergency"]))
    number_inpatient = st.number_input("Inpatient visits (prev year)", 0, 50, int(v["number_inpatient"]))

with col2:
    st.markdown("**Patient demographics**")
    age_encoded = st.selectbox("Age group (midpoint)", [5,15,25,35,45,55,65,75,85,95],
                               index=[5,15,25,35,45,55,65,75,85,95].index(int(v["age_encoded"])))
    gender_encoded = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female",
                                  index=int(v["gender_encoded"]))
    race_encoded = st.selectbox("Race",
                                [0,1,2,3,4],
                                format_func=lambda x: ["Caucasian","AfricanAmerican","Hispanic","Asian","Other"][x],
                                index=int(v["race_encoded"]))
    admission_type_id = st.number_input("Admission type ID", 1, 8, int(v["admission_type_id"]))
    discharge_disposition_id = st.number_input("Discharge disposition ID", 1, 30, int(v["discharge_disposition_id"]))
    admission_source_id = st.number_input("Admission source ID", 1, 26, int(v["admission_source_id"]))

    st.markdown("**Lab results**")
    a1cresult_encoded = st.selectbox("HbA1c result",
                                     [0,1,2,3],
                                     format_func=lambda x: ["None","Normal",">7",">8"][x],
                                     index=int(v["a1cresult_encoded"]))
    max_glu_serum_encoded = st.selectbox("Max glucose serum",
                                         [0,1,2,3],
                                         format_func=lambda x: ["None","Normal",">200",">300"][x],
                                         index=int(v["max_glu_serum_encoded"]))

with col3:
    st.markdown("**Medications** (0=No, 1=Steady, 2=Up, 3=Down)")
    med_map = lambda x: ["No","Steady","Up","Down"][int(x)]
    metformin_encoded = st.selectbox("Metformin", [0,1,2,3], format_func=med_map, index=int(v["metformin_encoded"]))
    insulin_encoded = st.selectbox("Insulin", [0,1,2,3], format_func=med_map, index=int(v["insulin_encoded"]))
    glipizide_encoded = st.selectbox("Glipizide", [0,1,2,3], format_func=med_map, index=int(v["glipizide_encoded"]))
    glyburide_encoded = st.selectbox("Glyburide", [0,1,2,3], format_func=med_map, index=int(v["glyburide_encoded"]))
    pioglitazone_encoded = st.selectbox("Pioglitazone", [0,1,2,3], format_func=med_map, index=int(v["pioglitazone_encoded"]))
    rosiglitazone_encoded = st.selectbox("Rosiglitazone", [0,1,2,3], format_func=med_map, index=int(v["rosiglitazone_encoded"]))

    st.markdown("**Other**")
    change_encoded = st.selectbox("Medication change", [0,1], format_func=lambda x: "No" if x==0 else "Yes",
                                  index=int(v["change_encoded"]))
    diabetesmed_encoded = st.selectbox("Diabetes medication prescribed", [0,1],
                                       format_func=lambda x: "No" if x==0 else "Yes",
                                       index=int(v["diabetesmed_encoded"]))

    st.markdown("**Diagnosis codes**")
    diag_1_code = st.number_input("Diag 1 (numeric)", 0.0, 999.0, float(v["diag_1_code"]))
    diag_2_code = st.number_input("Diag 2 (numeric)", 0.0, 999.0, float(v["diag_2_code"]))
    diag_3_code = st.number_input("Diag 3 (numeric)", 0.0, 999.0, float(v["diag_3_code"]))

# ── Submit ───────────────────────────────────
st.divider()
if st.button("🔮 Predict Readmission", type="primary", use_container_width=True):
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

    with st.spinner("Calling API..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

            if resp.status_code == 200:
                result = resp.json()
                st.divider()
                if result["prediction"] == 1:
                    st.error("⚠️ HIGH RISK: Patient likely to be readmitted within 30 days")
                else:
                    st.success("✅ LOW RISK: Patient not likely to be readmitted within 30 days")

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Prediction", result["prediction_label"].replace("_", " "))
                with col_r2:
                    st.metric("Confidence", f"{result['probability']:.1%}")
                with col_r3:
                    st.metric("Response time", f"{result['response_time_ms']:.1f} ms")

                with st.expander("Model details"):
                    st.write(f"**Model:** {result['model_name']}")
                    st.write(f"**Version:** {result['model_version']}")
                    st.write(f"**Alias:** {result['model_alias']}")
                    st.write(f"**Request ID:** `{result['request_id']}`")

            elif resp.status_code == 422:
                st.error("Validation error: check input values")
                st.json(resp.json())
            elif resp.status_code == 503:
                st.error("Model not loaded yet. Run the Airflow DAG first.")
            else:
                st.error(f"API returned status {resp.status_code}")
                st.write(resp.text)

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}. Is the service running?")
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

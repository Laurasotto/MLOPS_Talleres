# MLOps Proyecto 2 — Kubernetes

**Pontificia Universidad Javeriana · Grupo 4**  
Thomas Rivera & Laura Sotto · 2026-1

## Descripción

Arquitectura completa de MLOps desplegada en Kubernetes para predecir la readmisión hospitalaria temprana (<30 días) de pacientes diabéticos. Utiliza el dataset [Diabetes 130-US hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Namespace: mlops-p2              │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ PostgreSQL│    │  MinIO   │    │  MLflow  │                  │
│  │(raw+clean│    │(artifacts│    │ tracking │                  │
│  │ +infer.) │    │  store)  │    │ server   │                  │
│  └─────┬────┘    └─────┬────┘    └─────┬────┘                  │
│        │               │               │                        │
│  ┌─────▼───────────────▼───────────────▼────┐                  │
│  │              Apache Airflow               │                  │
│  │   validate → load → quality → preprocess  │                  │
│  │   → split → train → register → champion   │                  │
│  └───────────────────────────────────────────┘                  │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  ┌──────────┐   │
│  │ FastAPI  │    │Streamlit │    │  Locust  │  │Prometheus│   │
│  │/predict  │◄───│    UI    │    │  load    │  │ +Grafana │   │
│  │/metrics  │    │          │    │  test    │  │          │   │
│  └──────────┘    └──────────┘    └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Puertos de acceso (Docker Desktop Kubernetes → localhost)

| Servicio    | NodePort | URL                          | Credenciales           |
|-------------|----------|------------------------------|------------------------|
| Airflow     | 30808    | http://localhost:30808       | airflow / airflow      |
| MLflow      | 30500    | http://localhost:30500       | —                      |
| MinIO       | 30901    | http://localhost:30901       | minioadmin / minioadmin123 |
| API         | 30800    | http://localhost:30800/docs  | —                      |
| Streamlit   | 30851    | http://localhost:30851       | —                      |
| Locust      | 30889    | http://localhost:30889       | —                      |
| Prometheus  | 30909    | http://localhost:30909       | —                      |
| Grafana     | 30300    | http://localhost:30300       | admin / admin123       |

## Pre-requisitos

- Docker Desktop con Kubernetes habilitado
- `kubectl` configurado apuntando a Docker Desktop
- Cuenta en DockerHub (`thomasriverafonseca`)
- Dataset `Diabetes.csv` descargado (ver sección Dataset)

## Dataset

Descargar de Google Drive:

```bash
# Opción 1 — gdown
pip install gdown
gdown "https://drive.google.com/uc?id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC" -O Diabetes.csv

# Opción 2 — wget / curl (si el enlace es directo)
wget -O Diabetes.csv "https://docs.google.com/uc?export=download&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC"
```

## Despliegue paso a paso

### 1. Construir y publicar imágenes Docker

```bash
cd segundo_proyecto

# API de inferencia
docker build -t thomasriverafonseca/diabetes-api:latest ./api/
docker push thomasriverafonseca/diabetes-api:latest

# Streamlit UI
docker build -t thomasriverafonseca/diabetes-streamlit:latest ./streamlit_app/
docker push thomasriverafonseca/diabetes-streamlit:latest

# Airflow con dependencias ML
docker build -t thomasriverafonseca/diabetes-airflow:latest ./airflow/
docker push thomasriverafonseca/diabetes-airflow:latest

# Locust
docker build -t thomasriverafonseca/diabetes-locust:latest ./locust/
docker push thomasriverafonseca/diabetes-locust:latest
```

O usando el script automatizado:

```bash
./scripts/deploy.sh thomasriverafonseca
```

### 2. Aplicar manifiestos de Kubernetes

```bash
# Namespace
kubectl apply -f kubernetes/00-namespace/

# Infraestructura base
kubectl apply -f kubernetes/01-postgres/
kubectl apply -f kubernetes/02-minio/
kubectl apply -f kubernetes/03-mlflow/

# Esperar a que estén listos
kubectl rollout status statefulset/postgres -n mlops-p2 --timeout=120s
kubectl rollout status statefulset/minio    -n mlops-p2 --timeout=120s
kubectl rollout status deployment/mlflow   -n mlops-p2 --timeout=180s

# Observabilidad
kubectl apply -f kubernetes/08-prometheus/
kubectl apply -f kubernetes/09-grafana/

# Aplicaciones
kubectl apply -f kubernetes/04-airflow/
kubectl apply -f kubernetes/05-api/
kubectl apply -f kubernetes/06-streamlit/
kubectl apply -f kubernetes/07-locust/
```

### 3. Verificar el despliegue

```bash
kubectl get pods -n mlops-p2
kubectl get services -n mlops-p2
```

Todos los pods deben estar en estado `Running` o `Completed`.

### 4. Cargar el dataset en Airflow

```bash
# Copiar Diabetes.csv al PVC del scheduler
./scripts/copy_data.sh /ruta/a/Diabetes.csv

# O manualmente:
SCHED=$(kubectl get pod -n mlops-p2 -l app=airflow-scheduler -o jsonpath="{.items[0].metadata.name}")
kubectl cp Diabetes.csv mlops-p2/$SCHED:/opt/airflow/data/Diabetes.csv
```

También copiarlo al DAG PVC para el webserver:

```bash
WEB=$(kubectl get pod -n mlops-p2 -l app=airflow-webserver -o jsonpath="{.items[0].metadata.name}")
kubectl cp Diabetes.csv mlops-p2/$WEB:/opt/airflow/data/Diabetes.csv
```

### 5. Cargar el DAG en Airflow

```bash
WEB=$(kubectl get pod -n mlops-p2 -l app=airflow-webserver -o jsonpath="{.items[0].metadata.name}")
kubectl cp airflow/dags/dag_diabetes_pipeline.py mlops-p2/$WEB:/opt/airflow/dags/dag_diabetes_pipeline.py

SCHED=$(kubectl get pod -n mlops-p2 -l app=airflow-scheduler -o jsonpath="{.items[0].metadata.name}")
kubectl cp airflow/dags/dag_diabetes_pipeline.py mlops-p2/$SCHED:/opt/airflow/dags/dag_diabetes_pipeline.py
```

### 6. Ejecutar el pipeline

1. Abrir Airflow en http://localhost:30808 (airflow/airflow)
2. Habilitar el DAG `dag_diabetes_pipeline`
3. Hacer clic en **Trigger DAG** (botón ▶)
4. El DAG cargará un lote de 15,000 filas, las procesará, entrenará un RandomForest y lo promoverá como campeón en MLflow
5. Ejecutar el DAG múltiples veces para simular carga incremental (cada ejecución carga el siguiente lote)

### 7. Importar dashboard de Grafana

1. Abrir Grafana en http://localhost:30300 (admin/admin123)
2. Ir a **Dashboards → Import**
3. Subir `grafana/dashboards/diabetes_api_dashboard.json`

### 8. Prueba de carga con Locust

1. Abrir Locust en http://localhost:30889
2. Configurar: Number of users = 50, Spawn rate = 5, Host = http://api-service:8000
3. Iniciar prueba y observar métricas en Grafana en tiempo real

## Estructura del proyecto

```
segundo_proyecto/
├── airflow/
│   ├── dags/dag_diabetes_pipeline.py   # DAG principal (8 tareas)
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── main.py                         # FastAPI + Prometheus + DB logging
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit_app/
│   ├── app.py                          # UI de inferencia
│   ├── Dockerfile
│   └── requirements.txt
├── locust/
│   ├── locustfile.py                   # Escenario de carga
│   └── Dockerfile
├── db/
│   └── init.sql                        # Esquemas SQL completos
├── grafana/
│   └── dashboards/diabetes_api_dashboard.json
├── kubernetes/
│   ├── 00-namespace/
│   ├── 01-postgres/     (StatefulSet + Service + PVC + Secret + ConfigMap)
│   ├── 02-minio/        (StatefulSet + Service + PVC + Secret + ConfigMap)
│   ├── 03-mlflow/       (Deployment + Service + ConfigMap)
│   ├── 04-airflow/      (Deployment webserver+scheduler + Service + PVC)
│   ├── 05-api/          (Deployment 2 réplicas + Service + ConfigMap)
│   ├── 06-streamlit/    (Deployment + Service + ConfigMap)
│   ├── 07-locust/       (Deployment + Service + ConfigMap)
│   ├── 08-prometheus/   (Deployment + Service + RBAC + ConfigMap + PVC)
│   └── 09-grafana/      (Deployment + Service + Secret + ConfigMap + PVC)
└── scripts/
    ├── deploy.sh         # Build + push + apply todo
    ├── copy_data.sh      # Copiar CSV al PVC de Airflow
    └── teardown.sh       # Eliminar namespace completo
```

## Decisiones técnicas

### Métrica de selección de modelo: `f1_weighted`

Se eligió F1-score ponderado (weighted) como métrica principal porque:
- El dataset tiene desequilibrio de clases (solo ~11% de readmisiones <30 días)
- En contexto clínico, falsos negativos (no detectar una readmisión temprana) tienen mayor costo
- F1 weighted balancea precision y recall considerando el soporte de cada clase

### Carga incremental de datos

Cada ejecución del DAG carga exactamente 15,000 filas nuevas desde `Diabetes.csv`. El estado se persiste en `raw_data.batch_state`, garantizando que los datos no se dupliquen (via `ON CONFLICT DO NOTHING` con hash MD5).

### Estrategia de carga del modelo en la API

El modelo se carga una vez al inicio (`lifespan`) y se cachea en memoria (`_model_cache`). Para refrescar el modelo sin reiniciar el pod, existe el endpoint `POST /reload-model`. Esto evita la latencia de descargar el modelo desde MLflow en cada petición.

### Separación de capas de datos

| Capa | Tabla | Propósito |
|------|-------|-----------|
| RAW | `raw_data.diabetes_raw` | Datos originales inmutables, con metadata de lote |
| CLEAN | `clean_data.diabetes_clean` | Datos preprocesados listos para entrenamiento |
| SPLITS | `clean_data.diabetes_{train,val,test}` | Particiones por versión de batch |
| INFERENCE | `inference_logs.predictions` | Registro de todas las predicciones |

## Limpieza

```bash
./scripts/teardown.sh
# O:
kubectl delete namespace mlops-p2
```

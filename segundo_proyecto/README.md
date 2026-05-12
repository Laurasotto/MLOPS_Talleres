# MLOps Proyecto 2 — Kubernetes

**Pontificia Universidad Javeriana · Grupo 4**
Thomas Rivera & Laura Sotto · 2026-1

## Qué hace este proyecto

Pipeline completo de MLOps desplegado en Kubernetes que predice si un paciente diabético será readmitido al hospital en menos de 30 días. Los datos llegan en lotes incrementales, se procesan, se entrena un modelo, se versiona en MLflow y se sirve vía API. Todo corre en el namespace `mlops-p2` de Kubernetes.

## Arquitectura

<img width="1147" height="758" alt="image" src="https://github.com/user-attachments/assets/721fe24c-4d0a-4b50-a0d6-7ab8a83c6206" />


## URLs y credenciales

| Servicio   | URL                         | Usuario      | Contraseña     |
|------------|-----------------------------|--------------|----------------|
| Airflow    | http://localhost:30808      | airflow      | airflow        |
| MLflow     | http://localhost:30500      | —            | —              |
| MinIO      | http://localhost:30901      | minioadmin   | minioadmin123  |
| API docs   | http://localhost:30800/docs | —            | —              |
| Streamlit  | http://localhost:30851      | —            | —              |
| Locust     | http://localhost:30889      | —            | —              |
| Prometheus | http://localhost:30909      | —            | —              |
| Grafana    | http://localhost:30300      | admin        | admin123       |

## Requisitos previos

- Docker Desktop con Kubernetes habilitado (Settings → Kubernetes → Enable Kubernetes)
- `kubectl` apuntando a Docker Desktop — verificar con `kubectl cluster-info`
- Cuenta en DockerHub con acceso a `thomasriverafonseca/`
- Dataset `Diabetes.csv` descargado (ver sección Dataset)

## Dataset

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC" -O Diabetes.csv
```

Es el dataset *Diabetes 130-US hospitals (1999-2008)*. Contiene registros clínicos de pacientes hospitalizados con diabetes. La duración de estadía en el dataset va de 1 a 14 días — ese es el rango real de los datos, no una restricción arbitraria.

## Despliegue

### 1. Construir y publicar imágenes Docker

```bash
cd segundo_proyecto

docker build -t thomasriverafonseca/diabetes-api:latest ./api/
docker push thomasriverafonseca/diabetes-api:latest

docker build -t thomasriverafonseca/diabetes-streamlit:latest ./streamlit_app/
docker push thomasriverafonseca/diabetes-streamlit:latest

docker build -t thomasriverafonseca/diabetes-airflow:latest ./airflow/
docker push thomasriverafonseca/diabetes-airflow:latest

docker build -t thomasriverafonseca/diabetes-locust:latest ./locust/
docker push thomasriverafonseca/diabetes-locust:latest
```

O con el script automatizado:

```bash
./scripts/deploy.sh thomasriverafonseca
```

### 2. Aplicar manifiestos de Kubernetes

```bash
kubectl apply -f kubernetes/00-namespace/

kubectl apply -f kubernetes/01-postgres/
kubectl apply -f kubernetes/02-minio/
kubectl apply -f kubernetes/03-mlflow/

# Esperar a que la infraestructura base esté lista antes de continuar
kubectl rollout status statefulset/postgres -n mlops-p2 --timeout=120s
kubectl rollout status statefulset/minio    -n mlops-p2 --timeout=120s
kubectl rollout status deployment/mlflow   -n mlops-p2 --timeout=180s

kubectl apply -f kubernetes/08-prometheus/
kubectl apply -f kubernetes/09-grafana/

kubectl apply -f kubernetes/04-airflow/
kubectl apply -f kubernetes/05-api/
kubectl apply -f kubernetes/06-streamlit/
kubectl apply -f kubernetes/07-locust/
```

### 3. Verificar que todo corre

```bash
kubectl get pods -n mlops-p2
```

Todos los pods deben estar en `Running`. Si alguno no lo está:

```bash
kubectl describe pod <nombre-pod> -n mlops-p2
kubectl logs <nombre-pod> -n mlops-p2
```

### 4. Copiar el dataset a Airflow

```bash
./scripts/copy_data.sh /ruta/local/Diabetes.csv
```

O manualmente:

```bash
SCHED=$(kubectl get pod -n mlops-p2 -l app=airflow-scheduler -o jsonpath="{.items[0].metadata.name}")
kubectl cp Diabetes.csv mlops-p2/$SCHED:/opt/airflow/data/Diabetes.csv

WEB=$(kubectl get pod -n mlops-p2 -l app=airflow-webserver -o jsonpath="{.items[0].metadata.name}")
kubectl cp Diabetes.csv mlops-p2/$WEB:/opt/airflow/data/Diabetes.csv
```

### 5. Cargar el DAG en Airflow

```bash
WEB=$(kubectl get pod -n mlops-p2 -l app=airflow-webserver -o jsonpath="{.items[0].metadata.name}")
SCHED=$(kubectl get pod -n mlops-p2 -l app=airflow-scheduler -o jsonpath="{.items[0].metadata.name}")

kubectl cp airflow/dags/dag_diabetes_pipeline.py mlops-p2/$WEB:/opt/airflow/dags/dag_diabetes_pipeline.py
kubectl cp airflow/dags/dag_diabetes_pipeline.py mlops-p2/$SCHED:/opt/airflow/dags/dag_diabetes_pipeline.py

kubectl cp airflow/dags/dag_reset_pipeline.py mlops-p2/$WEB:/opt/airflow/dags/dag_reset_pipeline.py
kubectl cp airflow/dags/dag_reset_pipeline.py mlops-p2/$SCHED:/opt/airflow/dags/dag_reset_pipeline.py
```

### 6. Ejecutar el pipeline

1. Abrir Airflow en http://localhost:30808
2. Habilitar el DAG `dag_diabetes_pipeline`
3. Clic en **Trigger DAG** (botón de play)
4. Cada ejecución carga 15.000 filas nuevas, las procesa, entrena un RandomForest y lo registra en MLflow
5. La primera ejecución que complete `promote_champion` deja el modelo disponible para la API
6. Ejecutar varias veces para simular carga incremental — cada run toma el siguiente lote

Para resetear todo y empezar desde cero:

```bash
# Desde Airflow UI, trigger manualmente el DAG dag_reset_pipeline
# Esto trunca todas las tablas y resetea el experimento en MLflow
```

### 7. Importar dashboard en Grafana

```bash
DASHBOARD=$(cat grafana/dashboards/diabetes_api_dashboard.json)
curl -s -X POST "http://localhost:30300/api/dashboards/db" \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d "{\"dashboard\": $DASHBOARD, \"overwrite\": true, \"folderId\": 0}"
```

O desde la UI: Grafana → Dashboards → Import → subir `grafana/dashboards/diabetes_api_dashboard.json`.

### 8. Prueba de carga

1. Abrir Locust en http://localhost:30889
2. Number of users: `50`, Spawn rate: `5`, Host: `http://api-service:8000`
3. Start — y ver en Grafana cómo suben las métricas en tiempo real

## Estructura del proyecto

```
segundo_proyecto/
├── airflow/
│   ├── dags/
│   │   ├── dag_diabetes_pipeline.py   # DAG principal — 8 tareas
│   │   └── dag_reset_pipeline.py      # DAG para resetear tablas y MLflow
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── main.py                        # FastAPI + métricas Prometheus + logging DB
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit_app/
│   ├── app.py                         # UI de predicción en español
│   ├── Dockerfile
│   └── requirements.txt
├── locust/
│   ├── locustfile.py                  # Escenario de carga sobre /predict
│   └── Dockerfile
├── db/
│   └── init.sql                       # Creación de esquemas y tablas
├── grafana/
│   └── dashboards/
│       └── diabetes_api_dashboard.json
├── kubernetes/
│   ├── 00-namespace/
│   ├── 01-postgres/      StatefulSet + Service + PVC + Secret + ConfigMap
│   ├── 02-minio/         StatefulSet + Service + PVC + Secret + ConfigMap
│   ├── 03-mlflow/        Deployment + Service + ConfigMap
│   ├── 04-airflow/       Deployment webserver+scheduler + Service + PVCs
│   ├── 05-api/           Deployment 2 réplicas + Service + ConfigMap
│   ├── 06-streamlit/     Deployment + Service + ConfigMap
│   ├── 07-locust/        Deployment + Service + ConfigMap
│   ├── 08-prometheus/    Deployment + Service + RBAC + ConfigMap + PVC
│   └── 09-grafana/       Deployment + Service + Secret + ConfigMap + PVC
└── scripts/
    ├── deploy.sh          # Build + push + apply completo
    ├── copy_data.sh       # Copiar Diabetes.csv al PVC de Airflow
    └── teardown.sh        # Eliminar namespace y todos los recursos
```

## DAG — tareas y flujo

```
validate_source → load_batch → validate_quality → preprocess
    → split_data → train_model → register_mlflow → promote_champion
```

| Tarea | Qué hace |
|-------|----------|
| `validate_source` | Verifica que `Diabetes.csv` existe en el pod |
| `load_batch` | Carga las siguientes 15.000 filas en `raw_data.diabetes_raw` |
| `validate_quality` | Verifica nulos, tipos y rangos en los datos crudos |
| `preprocess` | Codifica variables categóricas y guarda en `clean_data.diabetes_clean` |
| `split_data` | Divide en train (70%) / val (15%) / test (15%) |
| `train_model` | Entrena RandomForestClassifier, evalúa con F1-weighted |
| `register_mlflow` | Registra parámetros, métricas, artefactos y modelo en MLflow |
| `promote_champion` | Compara F1 con el campeón anterior — si mejora, asigna alias `champion` |

## Tablas en PostgreSQL

| Schema | Tabla | Contenido |
|--------|-------|-----------|
| `raw_data` | `batch_state` | Estado de cada lote cargado (batch_id, filas, timestamp) |
| `raw_data` | `diabetes_raw` | Datos originales del CSV sin transformar |
| `clean_data` | `diabetes_clean` | Datos preprocesados y codificados |
| `clean_data` | `diabetes_train` | Split de entrenamiento (~70%) |
| `clean_data` | `diabetes_val` | Split de validación (~15%) |
| `clean_data` | `diabetes_test` | Split de prueba (~15%) |
| `inference_logs` | `predictions` | Registro de cada predicción: input, output, modelo, latencia |

Conectarse a postgres directamente:

```bash
kubectl exec -it postgres-0 -n mlops-p2 -- psql -U postgres -d mlops_db
```

## API — endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado de la API y conexión a MLflow |
| `/predict` | POST | Recibe 43 features clínicas, devuelve predicción + probabilidad |
| `/model-info` | GET | Nombre, versión y alias del modelo activo |
| `/reload-model` | POST | Recarga el modelo desde MLflow sin reiniciar el pod |
| `/metrics` | GET | Métricas en formato Prometheus |

La API tiene 2 réplicas para distribuir la carga de Locust. El modelo se carga una vez al iniciar y se cachea en memoria — no se descarga de MLflow en cada request.

## Modelo

- **Tipo:** RandomForestClassifier
- **Nombre en MLflow:** `diabetes_readmission_model`
- **Alias productivo:** `champion`
- **Métrica de selección:** F1-weighted — elegida porque el dataset tiene desequilibrio de clases (~11% de readmisiones <30 días) y accuracy sola sería engañosa
- **Variable objetivo:** readmisión hospitalaria en menos de 30 días (binaria)

## Observabilidad

Prometheus recolecta métricas desde `http://api-service:8000/metrics` cada 15 segundos. El dashboard de Grafana incluye:

- Total de requests
- Requests por segundo
- Latencia promedio
- Percentiles de latencia (p50, p95, p99)
- Tasa de error
- CPU real por pod (via cAdvisor)
- RAM real por pod (via cAdvisor)

Para verificar que Prometheus está recolectando correctamente, ir a http://localhost:30909 → Status → Targets. El target `diabetes-api` y `kubernetes-cadvisor` deben aparecer en `UP`.

## Imágenes Docker

Todas las imágenes propias están publicadas en DockerHub:

- `thomasriverafonseca/diabetes-api:latest`
- `thomasriverafonseca/diabetes-streamlit:latest`
- `thomasriverafonseca/diabetes-airflow:latest`
- `thomasriverafonseca/diabetes-locust:latest`

## Recursos por pod

| Pod / Deployment | Réplicas | CPU request | CPU limit | RAM request | RAM limit |
|------------------|----------|-------------|-----------|-------------|-----------|
| PostgreSQL | 1 | 250m | 1 | 512Mi | 1Gi |
| MinIO | 1 | 250m | 1 | 512Mi | 1Gi |
| MLflow | 1 | 250m | 1 | 512Mi | 1Gi |
| Airflow Webserver | 1 | 250m | 1 | 512Mi | 1Gi |
| Airflow Scheduler | 1 | 250m | 1 | 1Gi | 3Gi |
| FastAPI (api) | 2 | 200m | 1 | 512Mi | 1Gi |
| Streamlit | 1 | 100m | 500m | 256Mi | 512Mi |
| Locust | 1 | 100m | 500m | 256Mi | 512Mi |
| Prometheus | 1 | 100m | 500m | 256Mi | 512Mi |
| Grafana | 1 | 100m | 500m | 256Mi | 512Mi |

El Scheduler de Airflow tiene el límite de RAM más alto (3Gi) porque ejecuta las tareas del DAG en proceso — incluyendo el entrenamiento del RandomForest sobre los datos acumulados.

## Limpieza completa

```bash
kubectl delete namespace mlops-p2
```

Esto elimina todos los pods, servicios, PVCs y datos. Para volver a desplegar, seguir desde el paso 2.

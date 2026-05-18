#!/usr/bin/env bash
# Script de despliegue completo del proyecto en Kubernetes.
# Hace tres cosas en orden:
# 1) Construye y sube las imágenes Docker a DockerHub
# 2) Aplica los manifiestos de Kubernetes en el orden correcto (dependencias primero)
# 3) Espera a que los servicios críticos estén listos antes de continuar
#
# Uso: ./scripts/deploy.sh [DOCKERHUB_USER]
# Ejemplo: ./scripts/deploy.sh thomasriverafonseca

set -euo pipefail

DOCKER_USER="${1:-thomasriverafonseca}"  # usuario de DockerHub, con valor por defecto
K8S_DIR="$(cd "$(dirname "$0")/.." && pwd)/kubernetes"  # ruta absoluta a la carpeta kubernetes

echo "═══════════════════════════════════════════════"
echo "  MLOps Proyecto 2 — Kubernetes Deployment"
echo "  DockerHub user: $DOCKER_USER"
echo "═══════════════════════════════════════════════"

# ── Paso 1: Construir y subir imágenes Docker ─────────────────────────────
# Construimos una imagen por cada servicio que tiene su propio Dockerfile.
# La imagen se sube a DockerHub para que Kubernetes la pueda descargar desde los nodos.
echo ""
echo "▶ Building and pushing Docker images..."

for svc in api streamlit_app airflow locust; do
  # Mapeamos el nombre del directorio al nombre del tag de la imagen
  if [ "$svc" = "streamlit_app" ]; then
    tag="${DOCKER_USER}/diabetes-streamlit:latest"
  elif [ "$svc" = "airflow" ]; then
    tag="${DOCKER_USER}/diabetes-airflow:latest"
  else
    tag="${DOCKER_USER}/diabetes-${svc}:latest"
  fi
  echo "  Building $tag from ./$svc/..."
  docker build -t "$tag" "./$svc/"
  docker push "$tag"
done

# ── Paso 2: Aplicar manifiestos de Kubernetes en orden ────────────────────
# El orden importa porque hay dependencias entre servicios:
# postgres y minio deben estar listos antes de mlflow,
# y mlflow debe estar listo antes de airflow y la api.
echo ""
echo "▶ Applying Kubernetes manifests..."

apply_dir() {
  local dir="$1"
  echo "  Applying $dir..."
  kubectl apply -f "$dir/"
}

# Primero creamos el namespace, las bases de datos y el monitoreo
apply_dir "$K8S_DIR/00-namespace"
apply_dir "$K8S_DIR/01-postgres"
apply_dir "$K8S_DIR/02-minio"
apply_dir "$K8S_DIR/03-mlflow"
apply_dir "$K8S_DIR/08-prometheus"
apply_dir "$K8S_DIR/09-grafana"

# ── Paso 3: Esperar a que los servicios críticos estén listos ─────────────
# kubectl rollout status bloquea hasta que el deployment/statefulset está completamente listo.
# Esto garantiza que mlflow puede conectarse a postgres y minio antes de arrancar.
echo ""
echo "▶ Waiting for PostgreSQL to be ready..."
kubectl rollout status statefulset/postgres -n mlops-p2 --timeout=120s

echo "▶ Waiting for MinIO to be ready..."
kubectl rollout status statefulset/minio -n mlops-p2 --timeout=120s

echo "▶ Waiting for MLflow to be ready..."
kubectl rollout status deployment/mlflow -n mlops-p2 --timeout=180s

# Una vez que los servicios base están listos, desplegamos el resto
apply_dir "$K8S_DIR/04-airflow"
apply_dir "$K8S_DIR/05-api"
apply_dir "$K8S_DIR/06-streamlit"
apply_dir "$K8S_DIR/07-locust"

echo ""
echo "▶ Deployment complete. Checking pod status..."
kubectl get pods -n mlops-p2

# Imprimimos las URLs de acceso para facilitar el trabajo
echo ""
echo "═══════════════════════════════════════════════"
echo "  Access URLs (Docker Desktop Kubernetes):"
echo "  Airflow:    http://localhost:30808  (airflow/airflow)"
echo "  MLflow:     http://localhost:30500"
echo "  MinIO:      http://localhost:30901  (minioadmin/minioadmin123)"
echo "  API Docs:   http://localhost:30800/docs"
echo "  Streamlit:  http://localhost:30851"
echo "  Locust:     http://localhost:30889"
echo "  Prometheus: http://localhost:30909"
echo "  Grafana:    http://localhost:30300  (admin/admin123)"
echo "═══════════════════════════════════════════════"

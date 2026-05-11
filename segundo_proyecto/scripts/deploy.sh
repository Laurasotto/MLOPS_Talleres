#!/usr/bin/env bash
# Deploy all Kubernetes resources for MLOps Proyecto 2
# Usage: ./scripts/deploy.sh [DOCKERHUB_USER]

set -euo pipefail

DOCKER_USER="${1:-thomasriverafonseca}"
K8S_DIR="$(cd "$(dirname "$0")/.." && pwd)/kubernetes"

echo "═══════════════════════════════════════════════"
echo "  MLOps Proyecto 2 — Kubernetes Deployment"
echo "  DockerHub user: $DOCKER_USER"
echo "═══════════════════════════════════════════════"

# ── Build and push Docker images ─────────────────
echo ""
echo "▶ Building and pushing Docker images..."

for svc in api streamlit_app airflow locust; do
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

# ── Apply Kubernetes manifests in order ──────────
echo ""
echo "▶ Applying Kubernetes manifests..."

apply_dir() {
  local dir="$1"
  echo "  Applying $dir..."
  kubectl apply -f "$dir/"
}

apply_dir "$K8S_DIR/00-namespace"
apply_dir "$K8S_DIR/01-postgres"
apply_dir "$K8S_DIR/02-minio"
apply_dir "$K8S_DIR/03-mlflow"
apply_dir "$K8S_DIR/08-prometheus"
apply_dir "$K8S_DIR/09-grafana"

echo ""
echo "▶ Waiting for PostgreSQL to be ready..."
kubectl rollout status statefulset/postgres -n mlops-p2 --timeout=120s

echo "▶ Waiting for MinIO to be ready..."
kubectl rollout status statefulset/minio -n mlops-p2 --timeout=120s

echo "▶ Waiting for MLflow to be ready..."
kubectl rollout status deployment/mlflow -n mlops-p2 --timeout=180s

apply_dir "$K8S_DIR/04-airflow"
apply_dir "$K8S_DIR/05-api"
apply_dir "$K8S_DIR/06-streamlit"
apply_dir "$K8S_DIR/07-locust"

echo ""
echo "▶ Deployment complete. Checking pod status..."
kubectl get pods -n mlops-p2

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

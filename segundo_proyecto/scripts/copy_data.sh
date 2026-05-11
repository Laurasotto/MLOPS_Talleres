#!/usr/bin/env bash
# Copy the Diabetes dataset into the Airflow data PVC.
# Usage: ./scripts/copy_data.sh <path-to-Diabetes.csv>

set -euo pipefail

CSV_PATH="${1:-}"
if [ -z "$CSV_PATH" ]; then
  echo "Usage: $0 <path-to-Diabetes.csv>"
  exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
  echo "File not found: $CSV_PATH"
  exit 1
fi

NAMESPACE="mlops-p2"
SCHEDULER_POD=$(kubectl get pod -n "$NAMESPACE" -l app=airflow-scheduler -o jsonpath="{.items[0].metadata.name}")

echo "Copying $CSV_PATH → $SCHEDULER_POD:/opt/airflow/data/Diabetes.csv"
kubectl cp "$CSV_PATH" "$NAMESPACE/$SCHEDULER_POD:/opt/airflow/data/Diabetes.csv"
echo "Done."

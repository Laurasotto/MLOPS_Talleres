#!/usr/bin/env bash
# Copia el CSV del dataset de diabetes al volumen del scheduler de Airflow en Kubernetes.
# Lo hacemos así porque los PVCs no son accesibles directamente desde el host —
# la única forma de subir archivos a ellos es a través de un pod que los tenga montados.
#
# Uso: ./scripts/copy_data.sh <ruta-al-Diabetes.csv>
# Ejemplo: ./scripts/copy_data.sh ~/Downloads/Diabetes.csv

set -euo pipefail  # el script falla si cualquier comando falla, si una variable no está definida, o en pipes

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

# Obtenemos el nombre del pod del scheduler de Airflow dinámicamente —
# no lo hardcodeamos porque Kubernetes le asigna un sufijo aleatorio al nombre del pod.
SCHEDULER_POD=$(kubectl get pod -n "$NAMESPACE" -l app=airflow-scheduler -o jsonpath="{.items[0].metadata.name}")

echo "Copying $CSV_PATH → $SCHEDULER_POD:/opt/airflow/data/Diabetes.csv"

# kubectl cp funciona igual que scp — copia archivos hacia/desde pods.
# El formato es: namespace/pod:ruta-dentro-del-pod
kubectl cp "$CSV_PATH" "$NAMESPACE/$SCHEDULER_POD:/opt/airflow/data/Diabetes.csv"

echo "Done."

#!/usr/bin/env bash
# Elimina todos los recursos del proyecto de Kubernetes de una sola vez.
# Borrar el namespace elimina automáticamente todo lo que está dentro:
# pods, deployments, services, configmaps, secrets, PVCs, etc.
#
# ADVERTENCIA: esto también borra los PVCs y con ellos todos los datos
# de postgres, minio, prometheus y grafana. No hay forma de recuperarlos después.
# Solo ejecutar esto si queremos empezar desde cero.
set -euo pipefail

echo "Deleting namespace mlops-p2 (this will remove all resources)..."

# --ignore-not-found evita error si el namespace ya fue borrado antes
kubectl delete namespace mlops-p2 --ignore-not-found

echo "Done."

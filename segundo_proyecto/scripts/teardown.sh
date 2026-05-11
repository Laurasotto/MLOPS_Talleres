#!/usr/bin/env bash
# Remove all Kubernetes resources for MLOps Proyecto 2
set -euo pipefail

echo "Deleting namespace mlops-p2 (this will remove all resources)..."
kubectl delete namespace mlops-p2 --ignore-not-found

echo "Done."

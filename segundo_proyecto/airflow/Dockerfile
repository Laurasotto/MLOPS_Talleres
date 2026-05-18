# Usamos la imagen oficial de Airflow con Python 3.11 como base.
# Esta imagen ya trae Airflow instalado con sus dependencias — solo necesitamos
# agregar las librerías extras que usan nuestros DAGs.
FROM apache/airflow:2.9.2-python3.11

# Cambiamos temporalmente a root para instalar dependencias del sistema operativo.
# La imagen base de Airflow corre como usuario no-root por seguridad,
# pero apt-get requiere privilegios de root.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
# gcc y g++: compiladores necesarios para instalar algunas librerías Python con código C
# libpq-dev: cabeceras de PostgreSQL que necesita psycopg2 para compilarse

# Volvemos al usuario airflow antes de instalar las librerías Python —
# Airflow recomienda instalar paquetes como el usuario airflow, no root.
USER airflow

# Instalamos las dependencias Python extras que necesitan los DAGs:
# scikit-learn, mlflow, pandas, sqlalchemy, etc. (definidos en requirements.txt)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copiamos los DAGs dentro de la imagen.
# En Kubernetes, los DAGs también están en un PVC — esta copia sirve como respaldo inicial.
COPY dags/ /opt/airflow/dags/

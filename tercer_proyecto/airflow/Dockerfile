FROM apache/airflow:2.9.0

USER airflow

# Instalamos las dependencias del proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos los DAGs directamente en la imagen
COPY dags/ /opt/airflow/dags/

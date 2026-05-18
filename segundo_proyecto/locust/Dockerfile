# Usamos la imagen oficial de Locust como base — ya trae todo lo necesario para correr tests de carga.
# Solo necesitamos copiar nuestro archivo de escenarios (locustfile.py).
FROM locustio/locust:2.29.0

# Copiamos el locustfile al directorio de trabajo de Locust dentro del contenedor
COPY locustfile.py /home/locust/locustfile.py

WORKDIR /home/locust

# Puerto de la interfaz web de Locust — desde ahí configuramos cuántos usuarios simular
EXPOSE 8089

# Arrancamos Locust apuntando a nuestro locustfile.
# El host al que va a mandar tráfico se configura desde la UI o con --host en el deployment de Kubernetes.
CMD ["locust", "-f", "locustfile.py"]

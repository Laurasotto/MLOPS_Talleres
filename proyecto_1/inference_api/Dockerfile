# Imagen base: Python 3.11 en su version minimalista (slim)
# slim es mas liviana que la imagen completa de Python porque no incluye
# herramientas de compilacion ni librerias del sistema que no necesitamos
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
# Todos los comandos siguientes se ejecutaran desde aqui
WORKDIR /app

# Copiar el archivo de dependencias primero (antes del codigo)
# Esto aprovecha el cache de Docker: si requirements.txt no cambia,
# Docker no reinstala las dependencias en cada build
COPY requirements.txt /app/requirements.txt

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el codigo de la API al contenedor
COPY main.py /app/main.py

# Puerto en el que correra la API dentro del contenedor
EXPOSE 8084

# Comando para iniciar el servidor al arrancar el contenedor
# uvicorn es el servidor ASGI que corre FastAPI
# --host 0.0.0.0: acepta conexiones desde cualquier IP (necesario en Docker)
# --port 8084:    puerto dentro del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8084"]

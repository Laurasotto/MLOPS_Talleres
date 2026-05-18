# Usamos Python 3.11 en su versión "slim" — esto lo hacemos porque la imagen completa
# de Python pesa mucho más y no necesitamos todas las herramientas que trae.
# slim solo tiene lo mínimo para correr Python.
FROM python:3.11-slim

# Aquí definimos el directorio de trabajo dentro del contenedor.
# Todo lo que copiemos o ejecutemos a partir de aquí va a estar en /app.
WORKDIR /app

# Instalamos dependencias del sistema operativo que necesita nuestra app:
# - gcc: compilador que algunas librerías de Python necesitan para instalarse
# - libpq-dev: librería de PostgreSQL que necesita psycopg2 para conectarse a la BD
# Al final borramos el caché de apt para que la imagen quede lo más liviana posible.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el archivo de dependencias primero (antes que el código).
# Esto lo hacemos así porque Docker guarda en caché cada paso — si el código
# cambia pero requirements.txt no, Docker reutiliza la capa del pip install
# y no vuelve a descargar todas las librerías. Ahorra mucho tiempo en rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ahora sí copiamos el código de la API
COPY main.py .

# Le decimos a Docker que la aplicación va a escuchar en el puerto 8000.
# Esto es solo informativo — el puerto real se mapea en docker-compose.yaml.
EXPOSE 8000

# Comando que se ejecuta cuando el contenedor arranca.
# Usamos uvicorn como servidor ASGI para correr la app FastAPI definida en main.py.
# --host 0.0.0.0 hace que escuche en todas las interfaces de red (no solo localhost),
# lo cual es necesario para que otros servicios dentro de Docker puedan comunicarse con la API.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

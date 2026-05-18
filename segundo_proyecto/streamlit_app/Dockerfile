# Imagen de la interfaz web Streamlit — la UI que usan los usuarios para hacer predicciones.
# Usamos python:3.11-slim porque Streamlit es puro Python y no necesita dependencias del sistema.
FROM python:3.11-slim

WORKDIR /app

# Copiamos requirements primero para aprovechar el caché de Docker:
# si app.py cambia pero requirements.txt no, no volvemos a instalar todo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Puerto por defecto de Streamlit
EXPOSE 8501

# Arrancamos Streamlit apuntando a app.py.
# --server.address=0.0.0.0 es necesario para que otros contenedores puedan acceder a la app
# (sin esto solo escucha en localhost y no sería accesible desde fuera del contenedor).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

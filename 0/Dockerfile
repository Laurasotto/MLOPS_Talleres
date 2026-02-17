FROM python:3.11-slim

WORKDIR /app

COPY requisitos.txt .
RUN pip install --no-cache-dir -r requisitos.txt

COPY . .

# Entrena durante el build para que el modelo quede dentro de la imagen
RUN python modelo1.py

EXPOSE 8989

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8989"]


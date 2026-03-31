from locust import HttpUser, task, between
import random

class UsuarioDeCarga(HttpUser):
    # Tiempo de espera entre tareas por usuario simulado (en segundos)
    wait_time = between(1, 2.5)

    @task
    def hacer_inferencia(self):
        # Datos de ejemplo del dataset de diabetes (valores normalizados)
        payload = {
            "age":  round(random.uniform(-0.1, 0.1), 6),
            "sex":  round(random.choice([-0.044642, 0.050680]), 6),
            "bmi":  round(random.uniform(-0.09, 0.17), 6),
            "bp":   round(random.uniform(-0.07, 0.13), 6),
            "s1":   round(random.uniform(-0.12, 0.15), 6),
            "s2":   round(random.uniform(-0.11, 0.19), 6),
            "s3":   round(random.uniform(-0.10, 0.18), 6),
            "s4":   round(random.uniform(-0.07, 0.18), 6),
            "s5":   round(random.uniform(-0.12, 0.13), 6),
            "s6":   round(random.uniform(-0.13, 0.13), 6),
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error {response.status_code}: {response.text}")

    @task(1)
    def health_check(self):
        # También probamos el endpoint de health para monitorear disponibilidad
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check falló: {response.status_code}")

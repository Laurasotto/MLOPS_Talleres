# Pruebas de carga contra la API de inferencia del proyecto final.
# El pool de propiedades son registros reales del batch 1 del dataset,
# sacados directamente de clean.properties en Postgres antes de escribir este archivo.
# El 90% del tráfico va a /predict y el 10% a /health para simular
# los chequeos periódicos que haría un orquestador externo.

from locust import HttpUser, task, between
import random

# Registros reales del dataset, batch 1 (for_sale, varios estados de USA).
# Cada entrada es un diccionario listo para mandarse como payload al /predict.
PROPIEDADES = [
    {"bed": 4, "bath": 2, "acre_lot": 31.6,  "house_size": 2250, "status": "for_sale", "city": "Emmett",         "state": "Idaho",        "zip_code": "83617"},
    {"bed": 4, "bath": 2, "acre_lot": 2.35,  "house_size": 1988, "status": "for_sale", "city": "Lucedale",       "state": "Mississippi",  "zip_code": "39452"},
    {"bed": 3, "bath": 2, "acre_lot": 5.0,   "house_size": 1620, "status": "for_sale", "city": "Citra",          "state": "Florida",      "zip_code": "32113"},
    {"bed": 3, "bath": 2, "acre_lot": 0.48,  "house_size": 1560, "status": "for_sale", "city": "Zanesville",     "state": "Ohio",         "zip_code": "43701"},
    {"bed": 3, "bath": 1, "acre_lot": 0.26,  "house_size": 1562, "status": "for_sale", "city": "West Unity",     "state": "Ohio",         "zip_code": "43570"},
    {"bed": 3, "bath": 3, "acre_lot": 0.47,  "house_size": 2203, "status": "for_sale", "city": "Talladega",      "state": "Alabama",      "zip_code": "35160"},
    {"bed": 3, "bath": 2, "acre_lot": 0.12,  "house_size": 1600, "status": "for_sale", "city": "Columbus",       "state": "Ohio",         "zip_code": "43223"},
    {"bed": 4, "bath": 3, "acre_lot": 0.39,  "house_size": 2031, "status": "for_sale", "city": "Greenville",     "state": "Texas",        "zip_code": "75402"},
    {"bed": 4, "bath": 3, "acre_lot": 0.11,  "house_size": 1600, "status": "for_sale", "city": "Westchester",    "state": "Illinois",     "zip_code": "60154"},
    {"bed": 4, "bath": 1, "acre_lot": 1.77,  "house_size": 1520, "status": "for_sale", "city": "Windham",        "state": "Connecticut",  "zip_code": "6226"},
    {"bed": 3, "bath": 3, "acre_lot": 1.1,   "house_size": 2061, "status": "for_sale", "city": "Golden",         "state": "Missouri",     "zip_code": "65658"},
    {"bed": 2, "bath": 3, "acre_lot": 0.04,  "house_size": 1742, "status": "for_sale", "city": "Hiram",          "state": "Georgia",      "zip_code": "30141"},
    {"bed": 3, "bath": 1, "acre_lot": 0.15,  "house_size": 1522, "status": "for_sale", "city": "Akron",          "state": "Ohio",         "zip_code": "44312"},
    {"bed": 4, "bath": 3, "acre_lot": 0.19,  "house_size": 3367, "status": "for_sale", "city": "Normal",         "state": "Illinois",     "zip_code": "61761"},
    {"bed": 6, "bath": 2, "acre_lot": 0.08,  "house_size": 2934, "status": "for_sale", "city": "Schenectady",    "state": "New York",     "zip_code": "12308"},
    {"bed": 3, "bath": 3, "acre_lot": 0.93,  "house_size": 1530, "status": "for_sale", "city": "Whitehall",      "state": "Michigan",     "zip_code": "49461"},
    {"bed": 3, "bath": 3, "acre_lot": 0.18,  "house_size": 1510, "status": "for_sale", "city": "Miami Township", "state": "Ohio",         "zip_code": "45342"},
    {"bed": 5, "bath": 2, "acre_lot": 0.15,  "house_size": 1893, "status": "for_sale", "city": "Indianapolis",   "state": "Indiana",      "zip_code": "46237"},
    {"bed": 3, "bath": 1, "acre_lot": 0.22,  "house_size": 1572, "status": "for_sale", "city": "Roseville",      "state": "Minnesota",    "zip_code": "55113"},
    {"bed": 3, "bath": 1, "acre_lot": 0.26,  "house_size": 1680, "status": "for_sale", "city": "Morehead",       "state": "Kentucky",     "zip_code": "40351"},
]


class UsuarioInferencia(HttpUser):
    # Cada usuario virtual espera entre 0.5 y 2 segundos entre peticiones.
    wait_time = between(0.5, 2)

    @task(9)
    def predecir(self):
        """Manda una propiedad real del dataset y espera el precio estimado."""
        payload = random.choice(PROPIEDADES)
        # catch_response=True nos deja decidir manualmente si el request fue exitoso.
        # Si la API responde bien pero tarda más de 60 segundos lo marcamos fallido
        # porque en producción una predicción que demora tanto no es aceptable.
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Error {response.status_code}: {response.text}")
            elif response.elapsed.total_seconds() > 60:
                response.failure(f"Respuesta demasiado lenta: {response.elapsed.total_seconds():.1f}s")
            else:
                response.success()

    @task(1)
    def health_check(self):
        """Verifica que el servicio esté activo y con modelo cargado."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check falló: {response.status_code}")

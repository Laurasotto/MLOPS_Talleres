# Taller Locust — Pruebas de Carga

Este taller implementa pruebas de carga sobre la API de inferencia del taller de MLflow usando Locust. El objetivo es encontrar los recursos mínimos que soportan 10.000 usuarios concurrentes y analizar el comportamiento con múltiples réplicas.

## Prerequisitos

Este taller depende del taller de MLflow. Antes de levantar cualquier cosa aquí, el taller anterior debe estar corriendo porque la API de inferencia carga el modelo directamente desde MLflow.

Los siguientes contenedores deben estar activos:

- `mlflow_server` — servidor de MLflow en el puerto 5001
- `mlflow_minio` — almacenamiento de artefactos en el puerto 9002
- `mlflow_postgres` — base de datos en el puerto 5433

Para verificar que están corriendo:

```bash
docker ps | grep mlflow
```

Si no están activos, levantarlos desde la carpeta del taller anterior:

```bash
cd ../mlflow
docker compose up -d
```

Además, el modelo `diabetes_rf_regressor` debe estar registrado en MLflow. Si no se ha corrido el notebook del taller anterior, hacerlo antes de continuar.

## Arquitectura

La plataforma de pruebas está compuesta por tres componentes:

- **inference-api**: contenedor con la API de inferencia FastAPI publicada en DockerHub (`thomasriverafonseca/mlflow-api:latest`), que carga el modelo desde el MLflow del taller anterior.
- **nginx**: balanceador de carga que distribuye las peticiones entre las réplicas de la API cuando se usan múltiples instancias.
- **locust**: herramienta de pruebas de carga con interfaz web para configurar y monitorear los tests en tiempo real.

## Archivos

- `docker-compose.yaml` — levanta únicamente la API usando la imagen publicada en DockerHub, sin Locust.
- `docker-compose.locust.yaml` — levanta la API con recursos limitados, nginx y Locust para las pruebas de carga.
- `locustfile.py` — define el comportamiento de los usuarios simulados. Cada usuario hace peticiones POST a `/predict` con datos aleatorios del dataset de diabetes y GET a `/health`.
- `nginx.conf` — configuración del balanceador de carga para distribuir peticiones entre réplicas.

## Imagen en DockerHub

La imagen de inferencia está publicada en:

```
thomasriverafonseca/mlflow-api:latest
```

## Cómo ejecutar

Para levantar solo la API (sin pruebas de carga):

```bash
docker compose up --build
```

La API queda disponible en http://localhost:8001

Para levantar las pruebas de carga con Locust:

```bash
docker compose -f docker-compose.locust.yaml up --build
```

La UI de Locust queda disponible en http://localhost:8089. El host ya viene preconfigurado como `http://nginx:80`.

Para cambiar el número de réplicas de la API, modificar el valor de `replicas` en el `docker-compose.locust.yaml` y volver a levantar.

## Resultados de las pruebas

Todas las pruebas se realizaron con la API limitada a **0.5 CPU y 512MB de memoria** por réplica, agregando 500 usuarios por segundo hasta llegar al máximo configurado.

### 1 réplica

| Usuarios | RPS | Fallos | Promedio (ms) | P99 (ms) |
|----------|-----|--------|---------------|----------|
| 500      | 227 | 0%     | 174           | 700      |
| 1000     | 235 | 0%     | 1350          | 3600     |
| 2000     | 236 | 0%     | 3342          | 9700     |
| 5000     | 216 | 0%     | 7551          | 28000    |
| 10000    | 143 | 0%     | 9528          | 33000    |

### 3 réplicas con nginx

| Usuarios | RPS | Fallos | Promedio (ms) | P99 (ms) |
|----------|-----|--------|---------------|----------|
| 10000    | 158 | 0%     | 44896         | 80000    |

## Análisis

**¿Es posible reducir más los recursos?**

Con 0.5 CPU la API alcanza un techo de aproximadamente 235 RPS sin importar cuántos usuarios se agreguen. Reducir los recursos haría que ese techo bajara aún más y los tiempos de respuesta se volverían completamente inaceptables para cualquier uso real. No se recomienda bajar de 0.5 CPU para esta carga.

**¿Cuál es la mayor cantidad de peticiones soportadas?**

Con 1 réplica y 0.5 CPU el máximo alcanzado fue de **236 RPS** con 2000 usuarios. A partir de ahí el RPS empieza a bajar porque la cola de peticiones crece más rápido de lo que la API puede procesarlas, aunque sin generar errores.

**¿Qué diferencia hay entre una o múltiples instancias?**

Teóricamente múltiples réplicas deberían aumentar el throughput de forma proporcional al número de instancias. En esta prueba las 3 réplicas con nginx dieron peores resultados que una sola réplica porque el nginx agrega latencia adicional y las réplicas instalan dependencias al arrancar, lo que genera un cuello de botella diferente. En un entorno real con una imagen ya optimizada y sin instalaciones en tiempo de arranque, las réplicas sí mejorarían el rendimiento de forma significativa.

**¿Se logró llegar a 10.000 usuarios?**

Sí. La API soportó 10.000 usuarios concurrentes con 0% de fallos en todos los escenarios probados. Sin embargo, los tiempos de respuesta con esa carga son muy elevados (P99 de 33 segundos con 1 réplica), lo que indica que la API responde pero de forma lenta. Para tiempos de respuesta aceptables en producción se necesitarían más recursos por réplica o un mayor número de instancias con una imagen optimizada.

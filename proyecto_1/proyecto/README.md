# Proyecto 1 - MLOps: Orquestación, Entrenamiento y Modelos

**Pontificia Universidad Javeriana — Curso MLOps**  
**Grupo 4 — Thomas Rivera & Laura Sotto**

---

## ¿De qué trata este proyecto?

Este proyecto implementa un entorno completo de **MLOps** usando Docker Compose. La idea es construir un pipeline automatizado que recolecta datos de una API externa, los procesa, entrena un modelo de Machine Learning y lo expone a través de una API de inferencia, todo orquestado por Apache Airflow.

El dataset utilizado es una variante del **Forest Cover Type**, que contiene variables geográficas de terrenos boscosos (elevación, pendiente, tipo de suelo, etc.) y el objetivo es predecir el tipo de cobertura forestal (`cover_type` del 1 al 7).

---

## Arquitectura

El sistema está compuesto por los siguientes servicios corriendo en Docker Compose:

| Servicio | Descripción | Puerto |
|---|---|---|
| **Airflow** | Orquestador del pipeline | `8081` |
| **PostgreSQL** | Base de datos con 3 etapas de datos | `5432` |
| **Jupyter** | Entrenamiento del modelo | `8082` |
| **MinIO** | Almacenamiento de modelos | `8083` |
| **Inference API** | API de predicción (FastAPI) | `8084` |

<!-- Aquí puedes poner una imagen del diagrama de arquitectura -->

---

## Estructura del repositorio

```
proyecto_1/
├── P2/                              # API de datos del profesor
│   ├── main.py                      # API FastAPI que sirve los batches del dataset
│   ├── Dockerfile                   
│   ├── docker-compose.yaml          
│   ├── requirements.txt             
│   └── data/
│       └── covertype.csv            # Dataset Forest Cover Type
│
├── proyecto/                        # Pipeline MLOps principal
│   ├── docker-compose.yaml          # Orquestación de todos los servicios
│   ├── Dockerfile                   # Imagen de Airflow con dependencias
│   ├── Dockerfile.jupyter           # Imagen de Jupyter con dependencias
│   ├── requirements.txt             # Dependencias de Airflow
│   ├── requirements.jupyter.txt     # Dependencias de Jupyter
│   ├── dags/
│   │   ├── dag2.py                  # DAG principal del pipeline
│   │   └── dag_restart_data.py      # DAG utilitario para reiniciar datos
│   ├── notebooks/
│   │   └── train_model.ipynb        # Notebook de entrenamiento
│   └── logs/                        # Logs de Airflow (generados automaticamente)
│
└── inference_api/                   # API de inferencia
    ├── main.py                      # API FastAPI de prediccion
    ├── requirements.txt             # Dependencias de la API
    └── Dockerfile                   # Imagen de la API
```

---

## Requisitos previos

Antes de comenzar, asegúrate de tener instalado:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (versión reciente)
- [Docker Compose](https://docs.docker.com/compose/install/)

---

## Paso a paso: cómo levantar el sistema

### 1. Levantar la API de datos (P2)

Esta es la API del profesor que sirve los batches del dataset. Primero debes levantarla:

```bash
cd P2
docker-compose up -d
```

La API quedará disponible en `http://localhost:8080`.

### 2. Levantar el pipeline principal

```bash
cd ../proyecto
docker-compose up -d --build
```

Este comando construye las imágenes y levanta todos los contenedores. La primera vez puede tardar varios minutos porque descarga las imágenes base e instala las dependencias.

### 3. Verificar que todos los servicios estén corriendo

```bash
docker ps
```

Deberías ver activos: `postgres`, `airflow-webserver`, `airflow-scheduler`, `airflow-triggerer`, `jupyter`, `minio` y `inference-api`.

<!-- Aquí puedes poner una imagen del docker ps -->

---

## Interfaces gráficas

Una vez levantado el sistema, puedes acceder a las siguientes interfaces:

### Airflow — `http://localhost:8081`
Usuario: `airflow` | Contraseña: `airflow`

Aquí puedes ver y ejecutar los DAGs del pipeline.

<!-- Aquí puedes poner una imagen de la UI de Airflow con los DAGs -->

### Jupyter — `http://localhost:8082?token=mlops_token`

Aquí puedes ver y editar el notebook de entrenamiento `train_model.ipynb`.

<!-- Aquí puedes poner una imagen de Jupyter con el notebook -->

### MinIO — `http://localhost:8083`
Usuario: `minioadmin` | Contraseña: `minioadmin`

Aquí puedes ver los modelos entrenados almacenados en el bucket `models/group_4/`.

<!-- Aquí puedes poner una imagen de MinIO con los modelos -->

### Inference API — `http://localhost:8084/docs`

Aquí puedes probar la API de predicción usando el Swagger UI.

<!-- Aquí puedes poner una imagen del Swagger UI -->

---

## Cómo funciona el pipeline

El pipeline se ejecuta automáticamente cada 5 minutos gracias al scheduler de Airflow. Cada ejecución completa los siguientes pasos:

### Paso 1 — `create_tables`
Crea las tres tablas en PostgreSQL si no existen:
- `training_data`: datos crudos tal como vienen de la API
- `training_data_processed`: datos normalizados y con variables categóricas convertidas a números
- `training_data_ready`: datos limpios listos para entrenar, sin metadatos de pipeline

### Paso 2 — `get_api_data`
Llama a la API de datos (`http://host.docker.internal:8080/data`) enviando el número de grupo. La API devuelve un batch de datos del dataset Forest Cover Type.

<!-- Aquí puedes poner una imagen del log de get_api_data en Airflow -->

### Paso 3 — `preprocess_data`
Aplica tres transformaciones a los datos:
1. **Filtro de calidad**: elimina filas con `cover_type=0` (valores inválidos detectados en la API del profesor)
2. **Label Encoding**: convierte `wilderness_area` y `soil_type` de texto a número entero
3. **Normalización Min-Max**: escala todas las variables numéricas al rango [0, 1]

### Paso 4 — `trigger_jupyter`
Se comunica con Jupyter vía WebSocket para ejecutar el notebook `train_model.ipynb`. El notebook:
- Lee todos los batches acumulados de PostgreSQL (no solo el actual)
- Entrena un modelo Random Forest con `class_weight='balanced'` para manejar el desbalance de clases
- Guarda un bundle (modelo + encoders) en MinIO

<!-- Aquí puedes poner una imagen del log de trigger_jupyter mostrando el accuracy -->

### Paso 5 — `verify_and_stop`
Verifica que el modelo fue guardado correctamente en MinIO e incrementa un contador usando Airflow Variables. Cuando llega a 10 ejecuciones, pausa el DAG automáticamente.

<!-- Aquí puedes poner una imagen del grid de Airflow con las 10 ejecuciones verdes -->

---

## DAGs disponibles

### `dag_mlops_full_pipeline`
Pipeline principal. Se ejecuta cada 5 minutos y completa el proceso completo de ingesta, preprocesamiento, entrenamiento y almacenamiento. Se pausa automáticamente después de 10 ejecuciones exitosas.

### `dag_restart_data`
DAG utilitario de ejecución manual. Hace dos cosas:
1. Reinicia el contador de batches de la API externa (vuelve a batch 1)
2. Borra las tres tablas de PostgreSQL para empezar desde cero

Úsalo cuando quieras iniciar una nueva ronda de 10 batches.

---

## Modelos en MinIO

Cada ejecución del DAG genera un archivo `.pkl` en MinIO con el siguiente formato:

```
models/group_4/batch_{N}_acc_{accuracy}_{timestamp}.pkl
```

Por ejemplo:
```
models/group_4/batch_5_acc_0.8744_20260314_210709.pkl
```

El archivo contiene un bundle con tres componentes:
- `model`: el RandomForestClassifier entrenado
- `le_wilderness`: el LabelEncoder de wilderness_area
- `le_soil`: el LabelEncoder de soil_type

Guardar los encoders junto con el modelo garantiza que la inferencia use exactamente el mismo mapeo de categorías que se usó durante el entrenamiento.

<!-- Aquí puedes poner una imagen de MinIO con los modelos -->

---

## API de inferencia

La Inference API carga automáticamente el modelo más reciente de MinIO y expone tres endpoints:

### `GET /` — Health check
Verifica que la API está corriendo y que MinIO está conectado.

### `GET /models` — Listar modelos
Devuelve todos los modelos disponibles en MinIO ordenados por fecha.

### `POST /predict` — Predecir
Recibe los datos de un terreno y devuelve el tipo de cobertura forestal predicho.

**Ejemplo de request:**
```json
{
  "elevation": 2835,
  "aspect": 115,
  "slope": 26,
  "horizontal_distance_to_hydrology": 90,
  "vertical_distance_to_hydrology": 34,
  "horizontal_distance_to_roadways": 95,
  "hillshade_9am": 254,
  "hillshade_noon": 204,
  "hillshade_3pm": 61,
  "horizontal_distance_to_fire_points": 981,
  "wilderness_area": "Rawah",
  "soil_type": "C7746"
}
```

**Ejemplo de response:**
```json
{
  "cover_type": 1,
  "cover_type_name": "Spruce/Fir",
  "model_used": "group_4/batch_5_acc_0.8744_20260314_210709.pkl"
}
```

Si el `wilderness_area` o `soil_type` enviado no fue visto durante el entrenamiento, la API devuelve un error `400` con la lista de valores válidos.

<!-- Aquí puedes poner una imagen del Swagger UI con una predicción exitosa -->

---

## Tipos de cobertura forestal

| cover_type | Nombre |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## Bajar el sistema

```bash
cd proyecto
docker-compose down

cd ../P2
docker-compose down
```

Si quieres borrar también los volúmenes (datos de PostgreSQL y MinIO):

```bash
docker-compose down --volumes
```

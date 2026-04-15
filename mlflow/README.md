# MLflow Taller

Este taller implementa una plataforma completa de MLOps usando MLflow para el seguimiento de experimentos y registro de modelos. La arquitectura está compuesta por varios servicios que corren en Docker y se comunican entre sí a través de la red interna de docker-compose.

## Arquitectura

La plataforma usa los siguientes servicios:

- **PostgreSQL**: base de datos relacional que almacena los metadatos de MLflow y los datos del experimento (datos crudos y procesados).
- **MinIO**: almacenamiento de objetos compatible con S3 donde se guardan los artefactos del modelo.
- **MLflow Server**: servidor de tracking y model registry, conectado a PostgreSQL como backend y a MinIO para los artefactos.
- **JupyterLab**: entorno de experimentación donde se entrena el modelo y se registran los experimentos.
- **API de inferencia**: servicio FastAPI que carga el modelo desde MLflow y expone un endpoint de predicción.

Los servicios se comunican internamente usando los nombres de los contenedores como hostname. Desde el exterior se accede mediante los puertos expuestos en el host.

## Puertos

| Servicio    | Puerto local |
|-------------|-------------|
| PostgreSQL  | 5433        |
| MinIO API   | 9002        |
| MinIO UI    | 9003        |
| MLflow      | 5001        |
| JupyterLab  | 8890        |
| API         | 8000        |

Se usaron puertos distintos a los del proyecto anterior para evitar conflictos.

## Requisitos

- Docker Desktop instalado y corriendo
- Las carpetas `minio_data` y `postgres_data` se crean automáticamente al levantar los servicios

## Cómo levantar la plataforma

Clonar o ubicarse en la carpeta del proyecto y ejecutar:

```bash
docker compose up -d --build
```

Esto levanta todos los servicios. El bucket de MinIO se crea automáticamente gracias al servicio `minio_setup`. MLflow tarda aproximadamente un minuto en estar disponible porque instala sus dependencias al arrancar.

Para verificar que todo está corriendo:

```bash
docker compose ps
```

## Acceso a los servicios

- **MLflow UI**: http://localhost:5001
- **MinIO UI**: http://localhost:9003 — usuario: `admin`, contraseña: `supersecret`
- **JupyterLab**: http://localhost:8890 — el token se obtiene con `docker logs mlflow_jupyter 2>&1 | grep token`
- **API docs**: http://localhost:8000/docs


## PRUEBA
- docker logs mlflow_jupyter 2>&1 | grep token
 <img width="869" height="54" alt="image" src="https://github.com/user-attachments/assets/50581d95-5fd2-4b52-a055-ea5d4367532e" />
 <img width="1077" height="699" alt="image" src="https://github.com/user-attachments/assets/8aab5a6b-1535-4c22-a589-857220cc28c5" />

- Correr taller_mlflow.ipynb
  Importante tener en cuenta que en este paso se está trabajando con los hiperparametros: <img width="596" height="117" alt="image" src="https://github.com/user-attachments/assets/c01d6926-192e-43bd-903e-732781773151" />


- Ir a mlflow
  <img width="1370" height="539" alt="image" src="https://github.com/user-attachments/assets/f85e1d2e-3df2-4039-8b2b-fa4325c90dc5" />
 <img width="1487" height="332" alt="image" src="https://github.com/user-attachments/assets/9b511da4-70e1-4d9b-aff1-1c9b7c012a4e" />
 - Ir a la API
```bash
(http://localhost:8000/docs#/default/predict_predict_post)
```
<img width="1067" height="885" alt="image" src="https://github.com/user-attachments/assets/16b73e12-6d56-4978-9a0d-750a2251c5a9" />

   


## Experimentos

El notebook `taller_mlflow.ipynb` realiza el flujo completo:

1. Carga el dataset de diabetes de scikit-learn
2. Guarda los datos crudos en PostgreSQL en la tabla `diabetes_raw`
3. Normaliza los datos con StandardScaler y los guarda en la tabla `diabetes_processed`
4. Entrena un `RandomForestRegressor` usando `GridSearchCV` con 27 combinaciones de hiperparámetros (n_estimators, max_depth, max_features), todas registradas en MLflow
5. Registra el mejor modelo en el Model Registry de MLflow con el nombre `diabetes_rf_regressor`
6. Verifica que el modelo se puede cargar desde el registry y hacer predicciones

## API de inferencia

La API está construida con FastAPI y carga el modelo directamente desde MLflow al iniciar. Expone los siguientes endpoints:

- `GET /health` — verifica que la API está activa y el modelo cargado
- `POST /predict` — recibe los 10 features del dataset de diabetes y retorna la predicción
- `POST /predict/batch` — igual que el anterior pero acepta múltiples instancias

La documentación interactiva está disponible en http://localhost:8000/docs

## Apagar los servicios

```bash
docker compose down
```

Para borrar también los datos almacenados:

```bash
docker compose down -v
rm -rf ./postgres_data ./minio_data
```

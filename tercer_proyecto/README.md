# Proyecto Final MLOps — Nivel 4
**Pontificia Universidad Javeriana — Grupo 4**
Thomas Rivera & Laura Sotto

---

## Descripcion general

El proyecto consiste en construir un sistema MLOps completo con flujo automatizado de recoleccion de datos, validacion, procesamiento, entrenamiento, versionamiento, despliegue, inferencia y observabilidad. El problema de aprendizaje es regresion: estimar el precio de una propiedad inmobiliaria a partir de sus caracteristicas estructurales, geograficas y comerciales.

Todos los componentes se despliegan en Kubernetes, con imagenes construidas y publicadas en DockerHub mediante GitHub Actions, y sincronizadas con el cluster mediante Argo CD bajo un enfoque GitOps. Cada componente corre en su propio namespace de Kubernetes.

---

## Fuente de datos

Los datos se obtienen de una API externa provista por el docente. La imagen de Docker es `cristiandiaz13/mlops-puj:data-api-pf-v1`.

Para instanciarla localmente:

```bash
docker run --rm -p 8000:80 cristiandiaz13/mlops-puj:data-api-pf-v1
```

### Endpoints disponibles

| Endpoint | Metodo | Descripcion |
|---|---|---|
| `/health` | GET | Healthcheck de la API |
| `/data?group_number=4` | GET | Obtiene el siguiente batch de datos para el grupo |
| `/restart_data_generation?group_number=4` | GET | Reinicia la secuencia de batches al batch 0 |

Cada llamada a `/data` avanza al siguiente batch. Cuando se agotan todos los batches disponibles, la API responde con HTTP 400 y el mensaje `"Ya se recolecto toda la informacion minima necesaria"`. El cliente implementado en Airflow debe manejar ese caso explicitamente.

---

## Exploracion de la API

Realizamos una exploracion completa de la API para entender el volumen, estructura y comportamiento de los datos antes de disenar el pipeline. Consumimos todos los batches disponibles dos veces de forma independiente y confirmamos que los batches son deterministas: siempre devuelven exactamente los mismos registros en el mismo orden.

### Resultados por batch

| Batch | Registros | Status | Estados USA | Price min | Price max | Price promedio | Nulos |
|-------|----------:|--------|:-----------:|----------:|----------:|---------------:|-------|
| 0 | 73,784 | for_sale | 50 | $1 | $300,000 | $219,186 | ninguno |
| 1 | 94,551 | for_sale | 51 | $1 | $300,000 | $179,512 | ninguno |
| 2 | 230,366 | sold | 50 | $1 | $300,000 | $196,747 | ninguno |
| 3 | 4,055 | sold / for_sale | 1 | $10,000 | $300,000 | $236,810 | ninguno |
| 4 | 320,696 | sold / for_sale | 53 | $300,023 | $500,000 | $397,339 | ninguno |
| 5 | 361,457 | sold / for_sale | 53 | $500,007 | $515,000,000 | $1,134,678 | ninguno |

**Total: 1,084,909 registros en 6 batches (0 al 5). Ningun batch contiene nulos.**

### Observaciones identificadas

- **Batch 3** tiene solo 4,055 registros y cubre unicamente 1 estado de USA. Lo identificamos como un batch anomalo que el DAG deberia usar para ejercitar la decision de no entrenar por volumen insuficiente y falta de cobertura geografica.
- **Batch 4** introduce un cambio abrupto en la distribucion de precios: el precio minimo salta a $300,023 y el rango ya no se superpone con los batches anteriores. Esto constituye drift de distribucion detectable.
- **Batch 5** acentua ese drift con precios de hasta $515,000,000 y un promedio de mas de un millon de dolares. El modelo entrenado con los primeros batches no sera valido para este rango sin reentrenamiento.
- Los batches 0 y 1 solo tienen propiedades en venta (`for_sale`). El batch 2 tiene unicamente propiedades vendidas (`sold`). A partir del batch 3 se mezclan ambos estados.

Estas caracteristicas nos permiten anticipar el comportamiento del DAG en cada batch y disenar los criterios de decision de entrenamiento de forma justificada.

---

## Dataset

| Variable | Tipo | Descripcion |
|---|---|---|
| `price` | Numerica | **Variable objetivo.** Precio de cotizacion o venta de la propiedad |
| `bed` | Numerica | Numero de habitaciones |
| `bath` | Numerica | Numero de banos |
| `acre_lot` | Numerica | Tamano del terreno en acres |
| `house_size` | Numerica | Area habitable en pies cuadrados |
| `brokered_by` | Categorica | Agencia o corredor codificado |
| `status` | Categorica | Estado de la propiedad (`for_sale`, `sold`) |
| `street` | Categorica | Direccion codificada |
| `city` | Categorica | Ciudad |
| `state` | Categorica | Estado o region de USA |
| `zip_code` | Categorica | Codigo postal |
| `prev_sold_date` | Fecha | Fecha de venta anterior |

---

## Diseno de base de datos

Decidimos usar una sola instancia de PostgreSQL con tres schemas separados: `raw`, `clean` y `mlflow`. El schema de MLflow lo crea automaticamente el servidor al arrancar, no lo definimos nosotros.

### Schema raw

| Tabla | Descripcion |
|---|---|
| `raw.batches` | Metadata de cada batch ingestado desde la API |
| `raw.properties` | Registros crudos tal como llegan, sin ninguna modificacion |
| `raw.batch_audit` | Validaciones, decisiones y resultados por batch, es la fuente de datos de Streamlit |
| `raw.inference_events` | Registro de cada prediccion realizada por FastAPI |

### Schema clean

| Tabla | Descripcion |
|---|---|
| `clean.properties` | Datos transformados y listos para entrenamiento |

Decidimos botar `street` y `brokered_by` en el paso de limpieza porque son IDs numericos codificados sin significado semantico recuperable. Las variables categoricas restantes (`zip_code`, `city`, `state`, `status`) se guardan como texto. Estamos pensando en aplicar el encoding como parte del pipeline de sklearn durante el entrenamiento y guardar ese pipeline como artefacto en MLflow junto al modelo, para garantizar consistencia entre entrenamiento e inferencia.

El flujo es: el batch llega y se almacena en `raw.properties` sin tocar. Airflow aplica la limpieza basica y guarda el resultado en `clean.properties`. Cuando el DAG decide entrenar, la tarea de entrenamiento lee desde `clean.properties`, aplica el encoding y entrena el modelo.

---

## Estado actual

- [x] Exploracion y documentacion de la API del docente
- [ ] Diseno de manifiestos de Kubernetes por namespace
- [ ] Implementacion del DAG principal de Airflow
- [ ] Configuracion de MLflow con PostgreSQL y MinIO
- [ ] Implementacion de FastAPI con recarga de modelo desde MLflow
- [ ] Implementacion de Streamlit
- [ ] Configuracion de Prometheus y Grafana
- [ ] Workflows de GitHub Actions para construccion y publicacion de imagenes
- [ ] Configuracion de Argo CD para sincronizacion GitOps
- [ ] Pruebas de carga con Locust

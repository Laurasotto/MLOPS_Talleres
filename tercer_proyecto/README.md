# Proyecto Final MLOps — Nivel 4
**Pontificia Universidad Javeriana — Grupo 4**
Thomas Rivera & Laura Sotto

---

## Descripcion general

El proyecto consiste en construir un sistema MLOps completo con flujo automatizado de recoleccion de datos, validacion, procesamiento, entrenamiento, versionamiento, despliegue, inferencia y observabilidad. El problema de aprendizaje es regresion: estimar el precio de una propiedad inmobiliaria a partir de sus caracteristicas estructurales, geograficas y comerciales.

Todos los componentes se despliegan en Kubernetes, con imagenes construidas y publicadas en DockerHub mediante GitHub Actions, y sincronizadas con el cluster mediante Argo CD bajo un enfoque GitOps. Cada componente corre en su propio namespace de Kubernetes.

---

## Acceso a los servicios

Una vez levantado el sistema, todos los servicios son accesibles desde el navegador:

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow UI | http://localhost:30810 | `admin` / `admin123` |
| MLflow UI | http://localhost:30501 | sin autenticacion |
| MinIO Console | http://localhost:30902 | `minioadmin` / `minioadmin123` |
| FastAPI (inferencia) | http://localhost:30801/docs | — |
| Streamlit | http://localhost:30852 | — |
| Grafana | http://localhost:30301 | `admin` / `admin` |
| Prometheus | http://localhost:30910 | sin autenticacion |
| Locust | http://localhost:8089 | sin autenticacion |

---

## Iniciar el sistema

```bash
kubectl apply -f tercer_proyecto/k8s/postgres/
kubectl apply -f tercer_proyecto/k8s/minio/
kubectl apply -f tercer_proyecto/k8s/mlflow/
kubectl apply -f tercer_proyecto/k8s/airflow/
kubectl apply -f tercer_proyecto/k8s/inference/
kubectl apply -f tercer_proyecto/k8s/streamlit/
kubectl apply -f tercer_proyecto/k8s/observabilidad/ --recursive
```

Espera 2-3 minutos a que todos los pods arranquen. Luego entra a Airflow (http://localhost:30810) y activa el DAG `dag_mlops_pipeline` para empezar a consumir datos.

Para verificar que todo este corriendo:

```bash
kubectl get pods -A | grep -E "postgres|airflow|mlflow|minio|inference|streamlit|observabilidad"
```

---

## Componentes

### PostgreSQL

Es la base de datos central del sistema. Usamos una sola instancia con tres schemas separados: `raw` para los datos tal como llegan de la API, `clean` para los datos procesados y listos para entrenamiento, y `mlflow` que lo crea automaticamente el servidor de MLflow al arrancar. Elegimos PostgreSQL porque Airflow, MLflow y la API de inferencia ya lo soportan nativamente, y nos evita tener que correr multiples bases de datos.

### MinIO

Es el almacenamiento de objetos que usamos como backend de artefactos para MLflow. Funciona con la misma API que S3 de AWS, lo que significa que MLflow lo trata exactamente igual que si estuviera en la nube. Aqui se guardan los modelos entrenados, los pipelines de preprocesamiento y cualquier artefacto que MLflow registre durante un experimento. Tiene dos buckets: `mlflow-artifacts` para los artefactos del modelo registry y `airflow-logs` para los logs remotos de Airflow.

### MLflow

Es el corazon del ciclo de vida del modelo. Cada vez que Airflow entrena un modelo nuevo, lo registra en MLflow con sus metricas (MAE, RMSE, R²) y sus artefactos (el pipeline de sklearn completo). El Model Registry de MLflow es lo que nos permite tener el concepto de modelo en produccion: cuando un candidato supera al campeon actual, recibe el alias `production` y la API de inferencia lo carga automaticamente en el proximo reinicio. Usamos la version 2.13.0 porque encontramos problemas de compatibilidad con versiones mas recientes del cliente contra este servidor.

### Airflow

Es el orquestador del pipeline. Corre el DAG `dag_mlops_pipeline` que consume un batch de datos de la API del docente, lo valida, lo procesa, decide si tiene sentido entrenar un modelo nuevo, lo entrena, lo evalua y decide si debe reemplazar al que esta en produccion. Todo esto de forma automatica y con trazabilidad completa en la tabla `raw.batch_audit`. El scheduler y el webserver comparten un PVC para los logs, de forma que la UI de Airflow puede mostrar los logs de cada tarea directamente.

### FastAPI

Es la API de inferencia que queda expuesta al mundo. Carga al arrancar el modelo que tenga el alias `production` en el MLflow Model Registry y lo sirve a traves de un endpoint `POST /predict`. Cada prediccion queda registrada en la tabla `raw.inference_events` de PostgreSQL para trazabilidad. Tambien expone un endpoint `GET /metrics` en formato Prometheus para que Grafana pueda graficarlo, y un `POST /reload-model` por si se necesita forzar la recarga del modelo sin reiniciar el pod.

### Streamlit

Es el dashboard de monitoreo y prediccion manual. Tiene dos partes: una vista de auditoría que muestra el historial de batches procesados con sus decisiones de entrenamiento y metricas, leyendo directamente de `raw.batch_audit`; y un formulario de prediccion donde se pueden ingresar las caracteristicas de una propiedad y obtener el precio estimado llamando a la API de inferencia. Lo conectamos directamente a PostgreSQL para la parte de monitoreo y a FastAPI para las predicciones.

### Prometheus y Grafana

Son la capa de observabilidad del sistema. Prometheus hace scraping cada 5 segundos del endpoint `/metrics` de la API de inferencia y almacena las series de tiempo. Grafana lee esas series y las muestra en un dashboard preconfigurado con paneles para el volumen de predicciones por minuto, la latencia promedio y el percentil 95. Ambos corren en el namespace `observabilidad`.

Las metricas que expone la API son:
- `predict_requests_total` — contador de predicciones realizadas
- `predict_latency_seconds` — histograma de latencia por prediccion

### Locust

Es la herramienta de pruebas de carga. Corre fuera de Kubernetes (con Docker Compose) y genera trafico sintetico contra la API de inferencia para simular multiples usuarios haciendo predicciones al mismo tiempo. Nos sirve para verificar que la API aguanta carga razonable y para poblar las graficas de Prometheus y Grafana con datos reales. Para levantarlo:


### Argo CD

Es el componente GitOps del sistema. Sincroniza automaticamente el estado del cluster de Kubernetes con los manifiestos que estan en el repositorio de GitHub. Cuando se hace un push a `main` con cambios en los manifiestos de `k8s/`, Argo CD detecta la diferencia y aplica los cambios al cluster sin intervencion manual. Cada componente tiene su propia Application en Argo CD, lo que permite ver el estado de sincronizacion de cada namespace por separado.

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

Cada llamada a `/data` avanza al siguiente batch. Cuando se agotan todos los batches disponibles, la API responde con HTTP 400 y el mensaje `"Ya se recolecto toda la informacion minima necesaria"`. El cliente implementado en Airflow maneja ese caso explicitamente.

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

## Diseño de base de datos

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

Decidimos botar `street` y `brokered_by` en el paso de limpieza porque son IDs numericos codificados sin significado semantico recuperable. Las variables categoricas restantes (`zip_code`, `city`, `state`, `status`) se guardan como texto. El encoding se aplica como parte del pipeline de sklearn durante el entrenamiento y ese pipeline se guarda como artefacto en MLflow junto al modelo, para garantizar consistencia entre entrenamiento e inferencia.

El flujo es: el batch llega y se almacena en `raw.properties` sin tocar. Airflow aplica la limpieza basica y guarda el resultado en `clean.properties`. Cuando el DAG decide entrenar, la tarea de entrenamiento lee desde `clean.properties`, aplica el encoding y entrena el modelo.

---

## Pipeline DAG

El DAG principal se llama `dag_mlops_pipeline` y se ejecuta manualmente o por schedule. Esta compuesto por 16 tareas organizadas en dos bifurcaciones en secuencia: la primera decide si se entrena, y la segunda decide si el modelo entrenado reemplaza al que esta en produccion.

### Ingesta y validacion

Cada corrida comienza en `obtener_lote_api`, que consulta el endpoint `/data?group_number=4` de la API del docente y persiste el batch completo en `raw.batches` y `raw.properties`. La tarea retorna solo el `batch_id` (un entero) para evitar que el XCom serialice cientos de miles de registros — aprendimos esto por las malas cuando el scheduler se caia con OOMKill al intentar pasar 230k filas como JSON entre tareas.

Con el `batch_id` disponible, cuatro tareas corren en paralelo:

- `validar_columnas` comprueba que el batch traiga todas las columnas esperadas.
- `validar_calidad_datos` aplica tres checks sobre `raw.properties`: primero verifica que el batch tenga al menos 10,000 registros (batches muy pequenos no tienen suficientes datos para que el modelo aprenda bien), luego cuenta los estados de USA distintos y rechaza el batch si tiene menos de 10 (un modelo entrenado con datos de un solo estado no generaliza a nivel nacional), y finalmente verifica que no mas del 20% de los precios sean invalidos o cero.
- `detectar_nuevas_categorias` compara las categorias de `city`, `state`, `zip_code` y `status` del batch actual contra el historico para identificar valores que el modelo en produccion nunca vio.
- `detectar_drift` aplica el test KS sobre las variables numericas comparando el batch actual contra los anteriores.

### Decision de entrenamiento

`decidir_entrenamiento` consolida los resultados de las cuatro validaciones y elige entre `entrenar_modelo` y `omitir_entrenamiento`. La logica es: si cualquier validacion falla (schema invalido, calidad insuficiente, preprocesamiento fallido) se omite sin discusion. Si el batch pasa todas las validaciones pero no hay drift ni categorias nuevas, tambien se omite porque no hay evidencia de que el modelo deba actualizarse. Solo se entrena si se detecto drift estadistico o categorias nuevas, o si es el primer batch del pipeline y se necesita establecer una linea base.

Esto significa que el batch 3 del dataset (4,055 registros, un unico estado de USA) siempre sera skipeado por `validar_calidad_datos` antes de llegar a `decidir_entrenamiento`, lo cual es el comportamiento correcto dado que ese batch es claramente anomalo.

### Entrenamiento y evaluacion

`preprocesar_datos` aplica limpieza basica, descarta `street` y `brokered_by`, y escribe el resultado en `clean.properties` con el `batch_id` correspondiente.

`entrenar_modelo` lee exclusivamente los registros del batch actual desde `clean.properties` usando `WHERE batch_id = %s`. Decidimos entrenar por batch y no sobre el acumulado historico para que cada modelo sea comparable con los demas: si entrenamos sobre el acumulado, el modelo del batch 5 siempre tendra mas datos y ganara por volumen, no por calidad de los datos. Con esta estrategia, un batch grande tiene ventaja real solo si sus datos son mejores.

El modelo es un `RandomForestRegressor` con sklearn, y se guarda como artefacto en MLflow junto con el pipeline completo de preprocesamiento. `evaluar_modelo` calcula MAE, RMSE y R² sobre el 20% de test del mismo batch y los registra en el run de MLflow.

### Decision de promocion

`comparar_con_produccion` recupera las metricas del modelo que tiene el alias `production` en el MLflow Model Registry y las compara con el candidato. Para que un candidato reemplace al champion necesita mejorar el MAE en al menos 3% sin degradar el RMSE mas de 1%.

`decidir_promocion` bifurca hacia `promover_modelo` o `rechazar_modelo`. Si se promueve, el nuevo modelo recibe el alias `production` en el registry. Si no hay ningun modelo productivo todavia, el primer candidato valido se promueve automaticamente para establecer la linea base.

`registrar_resultado` cierra cada corrida escribiendo en `raw.batch_audit` la decision final, las metricas del candidato y el motivo de promocion o rechazo. Esta tabla es la fuente de verdad para el dashboard de Streamlit.

---

## Apagar todo

```bash
kubectl delete namespace airflow inference minio mlflow observabilidad streamlit postgres
cd tercer_proyecto/locust && docker compose down
```

**Importante:** al borrar el namespace `postgres` se pierden todos los datos — batches, modelos y registros de inferencia. Al volver a levantar hay que correr el DAG desde cero.

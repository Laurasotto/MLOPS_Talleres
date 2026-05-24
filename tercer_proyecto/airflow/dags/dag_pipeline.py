# DAG principal del pipeline MLOps
# Tareas: fetch_batch -> store_raw -> validate_schema -> validate_quality ->
#         detect_categories -> detect_drift -> preprocess -> decide_training ->
#         [train -> evaluate -> register -> compare -> decide_promotion ->
#          [promote | reject]] | skip_training -> log_result

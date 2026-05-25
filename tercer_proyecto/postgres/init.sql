-- =============================================================
-- Proyecto Final MLOps -- Grupo 4
-- Inicializacion de schemas y tablas
-- =============================================================

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;

-- =============================================================
-- SCHEMA RAW
-- =============================================================

CREATE TABLE IF NOT EXISTS raw.batches (
    id              SERIAL PRIMARY KEY,
    batch_number    INTEGER UNIQUE NOT NULL,
    fetched_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    record_count    INTEGER NOT NULL,
    fetch_status    VARCHAR(20) NOT NULL  -- success, error, no_more_data
);

CREATE TABLE IF NOT EXISTS raw.properties (
    id              SERIAL PRIMARY KEY,
    batch_id        INTEGER NOT NULL REFERENCES raw.batches(id),
    brokered_by     FLOAT,
    status          VARCHAR(20),
    price           FLOAT,
    bed             FLOAT,
    bath            FLOAT,
    acre_lot        FLOAT,
    street          FLOAT,
    city            VARCHAR(100),
    state           VARCHAR(100),
    zip_code        VARCHAR(20),
    house_size      FLOAT,
    prev_sold_date  VARCHAR(20),
    ingested_at     TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.batch_audit (
    id                  SERIAL PRIMARY KEY,
    batch_id            INTEGER NOT NULL REFERENCES raw.batches(id),
    dag_run_id          VARCHAR(200),
    executed_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    schema_valid        BOOLEAN,
    quality_valid       BOOLEAN,
    drift_detected      BOOLEAN,
    new_categories      JSONB,
    volume_sufficient   BOOLEAN,
    decision            VARCHAR(10),   -- train, skip
    decision_reason     TEXT,
    candidate_run_id    VARCHAR(200),
    candidate_mae       FLOAT,
    candidate_rmse      FLOAT,
    candidate_r2        FLOAT,
    promoted            BOOLEAN,
    promotion_reason    TEXT
);

CREATE TABLE IF NOT EXISTS raw.inference_events (
    id              SERIAL PRIMARY KEY,
    requested_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    input_data      JSONB,
    prediction      FLOAT,
    model_version   VARCHAR(100),
    model_alias     VARCHAR(100),
    response_status VARCHAR(20),  -- success, error
    error_message   TEXT
);

-- =============================================================
-- SCHEMA CLEAN
-- =============================================================

CREATE TABLE IF NOT EXISTS clean.properties (
    id              SERIAL PRIMARY KEY,
    batch_id        INTEGER NOT NULL REFERENCES raw.batches(id),
    price           FLOAT NOT NULL,
    bed             FLOAT,
    bath            FLOAT,
    acre_lot        FLOAT,
    house_size      FLOAT,
    status          VARCHAR(20),
    city            VARCHAR(100),
    state           VARCHAR(100),
    zip_code        VARCHAR(20),
    prev_sold_date  VARCHAR(20),
    processed_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

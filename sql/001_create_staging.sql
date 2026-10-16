
CREATE TABLE IF NOT EXISTS public.ingestion_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    source_year INTEGER,
    source_month INTEGER,
    status TEXT NOT NULL
        CHECK (status IN ('running', 'succeeded', 'failed')),
    records_seen BIGINT NOT NULL DEFAULT 0,
    records_inserted BIGINT NOT NULL DEFAULT 0,
    records_skipped BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    error_message TEXT,
    CONSTRAINT ck_ingestion_runs_source_period
        CHECK (
            (source_year IS NULL AND source_month IS NULL)
            OR (
                source_year IS NOT NULL
                AND source_month IS NOT NULL
                AND source_year > 0
                AND source_month BETWEEN 1 AND 12
            )
        )
);

CREATE TABLE IF NOT EXISTS bts_flight_staging (
    source_sha256 CHAR(64) NOT NULL,
    source_row_number BIGINT NOT NULL CHECK (source_row_number > 0),
    ingestion_run_id BIGINT NOT NULL
        REFERENCES ingestion_runs(run_id),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    year_raw TEXT,
    quarter_raw TEXT,
    month_raw TEXT,
    day_of_month_raw TEXT,
    day_of_week_raw TEXT,
    flight_date_raw TEXT,
    op_unique_carrier_raw TEXT,
    origin_airport_id_raw TEXT,
    origin_raw TEXT,
    dest_airport_id_raw TEXT,
    dest_raw TEXT,
    crs_dep_time_raw TEXT,
    dep_del15_raw TEXT,
    crs_arr_time_raw TEXT,
    cancelled_raw TEXT,
    diverted_raw TEXT,
    crs_elapsed_time_raw TEXT,
    distance_raw TEXT,

    CONSTRAINT pk_bts_flight_staging
        PRIMARY KEY (source_sha256, source_row_number)
);

CREATE INDEX IF NOT EXISTS ix_bts_staging_flight_date
    ON bts_flight_staging (flight_date_raw);

CREATE INDEX IF NOT EXISTS ix_ingestion_runs_started_at
    ON ingestion_runs (started_at DESC);

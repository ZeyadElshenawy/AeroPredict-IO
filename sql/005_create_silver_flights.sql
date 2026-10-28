CREATE TABLE IF NOT EXISTS public.silver_flights (
    source_sha256 CHAR(64) NOT NULL,
    source_row_number BIGINT NOT NULL CHECK (source_row_number > 0),
    ingestion_run_id BIGINT NOT NULL
        REFERENCES public.ingestion_runs(run_id),
    ingested_at TIMESTAMPTZ NOT NULL,
    transformed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    flight_date DATE NOT NULL,
    year SMALLINT NOT NULL CHECK (year >= 1987),
    quarter SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    day_of_month SMALLINT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),

    op_unique_carrier TEXT NOT NULL CHECK (length(op_unique_carrier) > 0),
    origin_airport_id BIGINT NOT NULL,
    origin TEXT NOT NULL CHECK (length(origin) = 3),
    dest_airport_id BIGINT NOT NULL,
    dest TEXT NOT NULL CHECK (length(dest) = 3),

    crs_dep_time_raw TEXT NOT NULL,
    crs_arr_time_raw TEXT NOT NULL,
    crs_dep_time_str CHAR(4) NOT NULL,
    crs_arr_time_str CHAR(4) NOT NULL,
    crs_dep_time TIME WITHOUT TIME ZONE NOT NULL,
    crs_arr_time TIME WITHOUT TIME ZONE NOT NULL,
    crs_dep_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,

    crs_elapsed_time DOUBLE PRECISION,
    distance DOUBLE PRECISION NOT NULL CHECK (distance >= 0),

    cancelled BOOLEAN NOT NULL,
    diverted BOOLEAN NOT NULL,
    dep_del15 SMALLINT CHECK (dep_del15 IS NULL OR dep_del15 IN (0, 1)),
    is_training_eligible BOOLEAN NOT NULL,

    CONSTRAINT pk_silver_flights
        PRIMARY KEY (source_sha256, source_row_number),

    CONSTRAINT ck_silver_cancelled_dep_del15
        CHECK (cancelled = FALSE OR dep_del15 IS NULL),

    CONSTRAINT ck_silver_training_eligible
        CHECK (is_training_eligible = (dep_del15 IS NOT NULL AND cancelled = FALSE AND diverted = FALSE))
);

CREATE INDEX IF NOT EXISTS ix_silver_flights_flight_date
    ON public.silver_flights (flight_date);

CREATE INDEX IF NOT EXISTS ix_silver_flights_dep_timestamp
    ON public.silver_flights (crs_dep_timestamp);

CREATE INDEX IF NOT EXISTS ix_silver_flights_training_eligible
    ON public.silver_flights (flight_date)
    WHERE is_training_eligible = TRUE;

CREATE INDEX IF NOT EXISTS ix_silver_flights_origin_date
    ON public.silver_flights (origin, flight_date);

CREATE INDEX IF NOT EXISTS ix_silver_flights_carrier_date
    ON public.silver_flights (op_unique_carrier, flight_date);

CREATE INDEX IF NOT EXISTS ix_silver_flights_source_sha256
    ON public.silver_flights (source_sha256);

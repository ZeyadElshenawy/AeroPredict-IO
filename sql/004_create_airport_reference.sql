
CREATE TABLE IF NOT EXISTS public.airport_reference (
    airport_seq_id BIGINT PRIMARY KEY,
    airport_id BIGINT NOT NULL,
    airport_code TEXT NOT NULL,
    airport_name TEXT,
    display_airport_name TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    utc_local_time_variation TEXT,
    airport_start_date DATE NOT NULL,
    airport_thru_date DATE,
    is_closed BOOLEAN NOT NULL,
    is_latest BOOLEAN NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    source_row_number BIGINT NOT NULL
        CHECK (source_row_number > 0),
    ingestion_run_id BIGINT NOT NULL
        REFERENCES public.ingestion_runs(run_id),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_airport_reference_latitude
        CHECK (latitude IS NULL OR latitude BETWEEN -90 AND 90),

    CONSTRAINT ck_airport_reference_longitude
        CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),

    CONSTRAINT uq_airport_reference_source_row
        UNIQUE (source_sha256, source_row_number)
);

CREATE INDEX IF NOT EXISTS ix_airport_reference_airport_id
    ON public.airport_reference (airport_id);

CREATE INDEX IF NOT EXISTS ix_airport_reference_latest
    ON public.airport_reference (airport_id)
    WHERE is_latest = TRUE;

CREATE INDEX IF NOT EXISTS ix_airport_reference_source
    ON public.airport_reference (source_sha256);

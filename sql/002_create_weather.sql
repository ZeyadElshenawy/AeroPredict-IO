CREATE TABLE IF NOT EXISTS public.weather_observations (
    station TEXT NOT NULL,
    observation_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,

    source_sha256 CHAR(64) NOT NULL,
    source_row_number BIGINT NOT NULL CHECK (source_row_number > 0),
    ingestion_run_id BIGINT NOT NULL
        REFERENCES public.ingestion_runs(run_id),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    tmpf_raw TEXT,
    dwpf_raw TEXT,
    relh_raw TEXT,
    drct_raw TEXT,
    sknt_raw TEXT,
    p01i_raw TEXT,
    alti_raw TEXT,
    mslp_raw TEXT,
    vsby_raw TEXT,
    gust_raw TEXT,
    skyc1_raw TEXT,
    skyc2_raw TEXT,
    skyc3_raw TEXT,
    skyc4_raw TEXT,
    skyl1_raw TEXT,
    skyl2_raw TEXT,
    skyl3_raw TEXT,
    skyl4_raw TEXT,
    wxcodes_raw TEXT,
    metar_raw TEXT,

    CONSTRAINT pk_weather_observations
        PRIMARY KEY (station, observation_time)
);

CREATE INDEX IF NOT EXISTS ix_weather_observations_time
    ON public.weather_observations (observation_time);

CREATE INDEX IF NOT EXISTS ix_weather_observations_source
    ON public.weather_observations (source_sha256);

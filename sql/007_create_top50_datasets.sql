-- Migration 007: Create Top 50 Hubs reference table and modeling views
-- Provides Baseline (Schedule Only) and Challenger (Weather Augmented) feature sets
-- for all commercial flights originating at the Top 50 US hub airports across 2025.

-- 1. Reference table of Top 50 Origin Hubs
CREATE TABLE IF NOT EXISTS public.top_50_airports (
    origin_airport_id  BIGINT PRIMARY KEY,
    airport_code       TEXT NOT NULL UNIQUE,
    station_id         TEXT NOT NULL,
    station_name       TEXT NOT NULL,
    network            TEXT NOT NULL,
    iana_timezone      TEXT NOT NULL,
    train_flights      BIGINT NOT NULL,
    pct_of_train       DOUBLE PRECISION NOT NULL,
    distance_km        DOUBLE PRECISION NOT NULL
);

-- 2. Baseline Modeling View (Schedule, Carrier, Temporal, Spatial features only)
CREATE OR REPLACE VIEW public.vw_gold_top50_baseline AS
SELECT
    f.source_sha256,
    f.source_row_number,
    f.dataset_split,
    f.flight_date,
    f.dep_del15,

    -- Calendar & Temporal Predictors
    f.year,
    f.month,
    f.day_of_month,
    f.day_of_week,
    f.quarter,
    f.is_weekend,
    f.dep_hour,
    f.dep_minute,
    f.sin_dep_time,
    f.cos_dep_time,
    f.sin_day_of_week,
    f.cos_day_of_week,
    f.sin_month,
    f.cos_month,

    -- Carrier & Route Predictors
    f.op_unique_carrier,
    f.origin_airport_id,
    t.airport_code AS origin,
    f.dest_airport_id,
    f.distance,
    f.crs_elapsed_time_clean,

    -- Spatial Predictors
    f.origin_latitude,
    f.origin_longitude,
    f.dest_latitude,
    f.dest_longitude
FROM public.gold_features f
JOIN public.top_50_airports t ON f.origin_airport_id = t.origin_airport_id;

-- 3. Challenger Modeling View (Baseline features + 2-hour pre-departure terminal weather)
CREATE OR REPLACE VIEW public.vw_gold_top50_weather AS
WITH flights_with_cutoff AS (
    SELECT
        f.*,
        t.airport_code AS origin,
        t.station_id,
        t.iana_timezone,
        -- Point-in-time prediction cutoff: exactly 2 hours prior to scheduled departure in UTC
        ((f.flight_date + (f.dep_hour || ' hours')::interval + (f.dep_minute || ' minutes')::interval)
         AT TIME ZONE t.iana_timezone) AT TIME ZONE 'UTC' - interval '2 hours' AS cutoff_utc
    FROM public.gold_features f
    JOIN public.top_50_airports t ON f.origin_airport_id = t.origin_airport_id
)
SELECT
    fc.source_sha256,
    fc.source_row_number,
    fc.dataset_split,
    fc.flight_date,
    fc.dep_del15,

    -- Calendar & Temporal Predictors
    fc.year,
    fc.month,
    fc.day_of_month,
    fc.day_of_week,
    fc.quarter,
    fc.is_weekend,
    fc.dep_hour,
    fc.dep_minute,
    fc.sin_dep_time,
    fc.cos_dep_time,
    fc.sin_day_of_week,
    fc.cos_day_of_week,
    fc.sin_month,
    fc.cos_month,

    -- Carrier & Route Predictors
    fc.op_unique_carrier,
    fc.origin_airport_id,
    fc.origin,
    fc.dest_airport_id,
    fc.distance,
    fc.crs_elapsed_time_clean,

    -- Spatial Predictors
    fc.origin_latitude,
    fc.origin_longitude,
    fc.dest_latitude,
    fc.dest_longitude,

    -- Point-in-Time Weather Predictors (T <= T_dep - 2 hours)
    w.observation_time AS weather_obs_time_utc,
    ROUND(EXTRACT(EPOCH FROM (fc.cutoff_utc - w.observation_time)) / 60.0, 1) AS weather_lag_minutes,
    NULLIF(w.tmpf_raw, 'M')::DOUBLE PRECISION AS weather_temp_f,
    NULLIF(w.dwpf_raw, 'M')::DOUBLE PRECISION AS weather_dew_point_f,
    NULLIF(w.relh_raw, 'M')::DOUBLE PRECISION AS weather_humidity_pct,
    COALESCE(NULLIF(w.drct_raw, 'M')::DOUBLE PRECISION, 0.0) AS weather_wind_dir_deg,
    COALESCE(NULLIF(w.sknt_raw, 'M')::DOUBLE PRECISION, 0.0) AS weather_wind_speed_kt,
    COALESCE(NULLIF(w.gust_raw, 'M')::DOUBLE PRECISION, 0.0) AS weather_wind_gust_kt,
    NULLIF(w.vsby_raw, 'M')::DOUBLE PRECISION AS weather_visibility_sm,
    NULLIF(w.alti_raw, 'M')::DOUBLE PRECISION AS weather_altimeter_in_hg,
    CASE
        WHEN w.p01i_raw = 'T' THEN 0.001
        WHEN w.p01i_raw IS NOT NULL AND w.p01i_raw ~ '^[0-9.]+$' THEN w.p01i_raw::DOUBLE PRECISION
        ELSE 0.0
    END AS weather_precip_1h_in,
    CASE
        WHEN (w.wxcodes_raw ~* '(RA|SN|DZ|PL|GS|GR|UP)' OR (w.p01i_raw IS NOT NULL AND w.p01i_raw NOT IN ('0.00', 'M', '0'))) THEN 1
        ELSE 0
    END AS weather_has_precip,
    CASE WHEN w.wxcodes_raw ~* 'TS' THEN 1 ELSE 0 END AS weather_has_thunderstorm,
    CASE WHEN w.wxcodes_raw ~* 'SN' THEN 1 ELSE 0 END AS weather_has_snow,
    CASE WHEN w.wxcodes_raw ~* '(FG|BR|HZ)' THEN 1 ELSE 0 END AS weather_has_fog_haze
FROM flights_with_cutoff fc
LEFT JOIN LATERAL (
    SELECT
        observation_time,
        tmpf_raw,
        dwpf_raw,
        relh_raw,
        drct_raw,
        sknt_raw,
        gust_raw,
        vsby_raw,
        alti_raw,
        p01i_raw,
        wxcodes_raw
    FROM public.weather_observations
    WHERE station = fc.station_id
      AND observation_time <= fc.cutoff_utc
      AND observation_time >= fc.cutoff_utc - interval '6 hours'
    ORDER BY observation_time DESC
    LIMIT 1
) w ON true;

-- Migration 006: Create partitioned Gold features table and split partitions
-- Follows repository PostgreSQL conventions; strictly separates target from predictors.
-- Invalid and nonpositive scheduled elapsed times are excluded via CHECK constraint.

CREATE TABLE public.gold_features (
    -- 1. Provenance & Partitioning Identifiers
    source_sha256            CHAR(64)          NOT NULL,
    source_row_number        BIGINT            NOT NULL CHECK (source_row_number > 0),
    flight_date              DATE              NOT NULL,
    dataset_split            TEXT              NOT NULL CHECK (dataset_split IN ('train', 'validation', 'test')),

    -- 2. Supervised Target (Strictly separated; never used as a predictor)
    dep_del15                SMALLINT          NOT NULL CHECK (dep_del15 IN (0, 1)),

    -- 3. Temporal & Calendar Predictors (Derived strictly from scheduled quantities)
    year                     SMALLINT          NOT NULL CHECK (year >= 1987),
    month                    SMALLINT          NOT NULL CHECK (month BETWEEN 1 AND 12),
    day_of_month             SMALLINT          NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_week              SMALLINT          NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),
    quarter                  SMALLINT          NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    is_weekend               BOOLEAN           NOT NULL,
    dep_hour                 SMALLINT          NOT NULL CHECK (dep_hour BETWEEN 0 AND 23),
    dep_minute               SMALLINT          NOT NULL CHECK (dep_minute BETWEEN 0 AND 59),
    sin_dep_time             DOUBLE PRECISION  NOT NULL CHECK (sin_dep_time BETWEEN -1.000001 AND 1.000001),
    cos_dep_time             DOUBLE PRECISION  NOT NULL CHECK (cos_dep_time BETWEEN -1.000001 AND 1.000001),
    sin_day_of_week          DOUBLE PRECISION  NOT NULL CHECK (sin_day_of_week BETWEEN -1.000001 AND 1.000001),
    cos_day_of_week          DOUBLE PRECISION  NOT NULL CHECK (cos_day_of_week BETWEEN -1.000001 AND 1.000001),
    sin_month                DOUBLE PRECISION  NOT NULL CHECK (sin_month BETWEEN -1.000001 AND 1.000001),
    cos_month                DOUBLE PRECISION  NOT NULL CHECK (cos_month BETWEEN -1.000001 AND 1.000001),

    -- 4. Carrier & Route Predictors
    op_unique_carrier        TEXT              NOT NULL CHECK (length(op_unique_carrier) > 0),
    origin_airport_id        BIGINT            NOT NULL,
    dest_airport_id          BIGINT            NOT NULL,
    distance                 DOUBLE PRECISION  NOT NULL CHECK (distance >= 0),
    crs_elapsed_time_clean   DOUBLE PRECISION  NOT NULL CHECK (crs_elapsed_time_clean > 0),

    -- 5. Verified Historical Spatial Predictors (from airport_reference)
    origin_latitude          DOUBLE PRECISION  NOT NULL CHECK (origin_latitude BETWEEN -90 AND 90),
    origin_longitude         DOUBLE PRECISION  NOT NULL CHECK (origin_longitude BETWEEN -180 AND 180),
    dest_latitude            DOUBLE PRECISION  NOT NULL CHECK (dest_latitude BETWEEN -90 AND 90),
    dest_longitude           DOUBLE PRECISION  NOT NULL CHECK (dest_longitude BETWEEN -180 AND 180),

    CONSTRAINT pk_gold_features PRIMARY KEY (source_sha256, source_row_number, dataset_split)
) PARTITION BY LIST (dataset_split);

-- Child Partition Tables
CREATE TABLE public.gold_features_train
    PARTITION OF public.gold_features FOR VALUES IN ('train');

CREATE TABLE public.gold_features_val
    PARTITION OF public.gold_features FOR VALUES IN ('validation');

CREATE TABLE public.gold_features_test
    PARTITION OF public.gold_features FOR VALUES IN ('test');

-- Partition-Pruned B-Tree Indexes
CREATE INDEX ix_gold_train_flight_date
    ON public.gold_features_train (flight_date);

CREATE INDEX ix_gold_train_carrier_date
    ON public.gold_features_train (op_unique_carrier, flight_date);

CREATE INDEX ix_gold_train_origin_date
    ON public.gold_features_train (origin_airport_id, flight_date);

CREATE INDEX ix_gold_val_flight_date
    ON public.gold_features_val (flight_date);

CREATE INDEX ix_gold_test_flight_date
    ON public.gold_features_test (flight_date);

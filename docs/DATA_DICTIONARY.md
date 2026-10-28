# FlightOps ML — Data Dictionary

> **Audit Status:** Verified against PostgreSQL container `flightops-ml-postgres-1` on October 20, 2026.
> **Schema:** `public` in PostgreSQL 17 database `flightops`.

---

## 1. Overview of Database Tables

| Table | Status | Verified Rows | Primary Key | Description |
|---|---|---:|---|---|
| [`public.ingestion_runs`](#2-table-publicingestion_runs) | Verified | 25 | `run_id` | Audit ledger tracking all batch file ingestion attempts and outcomes. |
| [`public.bts_flight_staging`](#3-table-publicbts_flight_staging-bronze) | Verified | 7,001,619 | (`source_sha256`, `source_row_number`) | Bronze untyped raw text staging for monthly BTS on-time flight files. |
| [`public.silver_flights`](#4-table-publicsilver_flights-silver) | Verified | 7,001,619 | (`source_sha256`, `source_row_number`) | Silver normalized, validated, and typed flight departure records. |
| [`public.airport_reference`](#5-table-publicairport_reference) | Verified | 20,295 | `airport_seq_id` | Master airport reference table containing coordinates and timezone variations. |
| [`public.weather_observations`](#6-table-publicweather_observations) | Verified (Test) | 8,760 | (`station`, `observation_time`) | ASOS surface weather observations (January 2025 test dataset for 10 stations). |
| [`public.gold_features`](#7-table-publicgold_features-gold-v1) | Verified | 6,879,484 | (`source_sha256`, `source_row_number`, `dataset_split`) | Curated model features with strict leakage prevention and split assignments. |

---

## 2. Table: `public.ingestion_runs`

Audit ledger recording every batch ingestion execution. Created by Migration 001, patched by Migration 003.

| Column | Type | Nullable | Default | Constraints / Description |
|---|---|---|---|---|
| `run_id` | `BIGINT` | `NOT NULL` | `GENERATED ALWAYS AS IDENTITY` | **Primary Key**. Unique surrogate run identifier. |
| `source_name` | `TEXT` | `NOT NULL` | | Identifier of data provider: `'BTS'`, `'IEM_ASOS'`, or `'BTS_AIRPORT_REFERENCE'`. |
| `source_filename` | `TEXT` | `NOT NULL` | | Original archive or file name (e.g. `'2025-01.zip'`, `'T_MASTER_CORD.csv'`). |
| `source_sha256` | `CHAR(64)` | `NOT NULL` | | Cryptographic SHA-256 hash of the ingested source file. |
| `source_year` | `INTEGER` | `NULL` | | Partition year (e.g. `2025`). Nullable for snapshot files (per Migration 003). |
| `source_month` | `INTEGER` | `NULL` | | Partition month (`1` to `12`). Nullable for snapshot files (per Migration 003). |
| `status` | `TEXT` | `NOT NULL` | | Execution status: `CHECK (status IN ('running', 'succeeded', 'failed'))`. |
| `records_seen` | `BIGINT` | `NOT NULL` | `0` | Total records scanned in source file. |
| `records_inserted` | `BIGINT` | `NOT NULL` | `0` | New records inserted into destination table. |
| `records_skipped` | `BIGINT` | `NOT NULL` | `0` | Duplicate records skipped (`ON CONFLICT DO NOTHING`). |
| `started_at` | `TIMESTAMPTZ` | `NOT NULL` | `NOW()` | Timestamp when the ingestion job commenced. |
| `finished_at` | `TIMESTAMPTZ` | `NULL` | | Timestamp when the ingestion job finished. |
| `error_message` | `TEXT` | `NULL` | | Truncated error traceback if `status = 'failed'`. |

**Check Constraints:**
- `ck_ingestion_runs_source_period`: Enforces that `source_year` and `source_month` are either both `NULL` (snapshots) or both valid integers (`year > 0`, `month BETWEEN 1 AND 12`).

---

## 3. Table: `public.bts_flight_staging` (Bronze)

Untyped staging table storing raw strings from monthly BTS reporting carrier archives. Created by Migration 001.

| Column | Type | Nullable | Source CSV Header | Description |
|---|---|---|---|---|
| `source_sha256` | `CHAR(64)` | `NOT NULL` | *(Derived)* | **PK Part 1**. SHA-256 fingerprint of the source ZIP file. |
| `source_row_number` | `BIGINT` | `NOT NULL` | *(Derived)* | **PK Part 2**. 1-based row index within the source CSV. `CHECK (source_row_number > 0)`. |
| `ingestion_run_id` | `BIGINT` | `NOT NULL` | *(Derived)* | **FK** $\rightarrow$ `public.ingestion_runs(run_id)`. |
| `ingested_at` | `TIMESTAMPTZ` | `NOT NULL` | *(Derived)* | System load timestamp (Default: `NOW()`). |
| `year_raw` | `TEXT` | `NULL` | `YEAR` | Scheduled departure year as formatted in CSV (e.g. `'2025.00'`). |
| `quarter_raw` | `TEXT` | `NULL` | `QUARTER` | Scheduled departure calendar quarter (e.g. `'1.00'`). |
| `month_raw` | `TEXT` | `NULL` | `MONTH` | Scheduled departure month (e.g. `'1.00'`). |
| `day_of_month_raw` | `TEXT` | `NULL` | `DAY_OF_MONTH` | Scheduled departure day of month (e.g. `'15.00'`). |
| `day_of_week_raw` | `TEXT` | `NULL` | `DAY_OF_WEEK` | Scheduled departure day of week (e.g. `'3.00'`). |
| `flight_date_raw` | `TEXT` | `NULL` | `FL_DATE` | Flight date string (e.g. `'1/15/2025 12:00:00 AM'` or `'2025-01-15'`). |
| `op_unique_carrier_raw` | `TEXT` | `NULL` | `OP_UNIQUE_CARRIER` | 2-letter DOT operating carrier code (e.g. `'AA'`, `'DL'`). |
| `origin_airport_id_raw` | `TEXT` | `NULL` | `ORIGIN_AIRPORT_ID` | BTS origin airport numeric ID (e.g. `'10397.00'`). |
| `origin_raw` | `TEXT` | `NULL` | `ORIGIN` | 3-letter IATA origin airport code (e.g. `'ATL'`). |
| `dest_airport_id_raw` | `TEXT` | `NULL` | `DEST_AIRPORT_ID` | BTS destination airport numeric ID (e.g. `'12892.00'`). |
| `dest_raw` | `TEXT` | `NULL` | `DEST` | 3-letter IATA destination airport code (e.g. `'LAX'`). |
| `crs_dep_time_raw` | `TEXT` | `NULL` | `CRS_DEP_TIME` | Scheduled departure clock time string (e.g. `'0922'`, `'2400'`). |
| `dep_del15_raw` | `TEXT` | `NULL` | `DEP_DEL15` | Raw departure delay indicator (`'0.00'`, `'1.00'`, or blank for cancelled). |
| `crs_arr_time_raw` | `TEXT` | `NULL` | `CRS_ARR_TIME` | Scheduled arrival clock time string (e.g. `'1145'`). |
| `cancelled_raw` | `TEXT` | `NULL` | `CANCELLED` | Cancellation indicator (`'0.00'`, `'1.00'`). |
| `diverted_raw` | `TEXT` | `NULL` | `DIVERTED` | Diversion indicator (`'0.00'`, `'1.00'`). |
| `crs_elapsed_time_raw` | `TEXT` | `NULL` | `CRS_ELAPSED_TIME` | Scheduled gate-to-gate elapsed time in minutes (e.g. `'143.00'`). |
| `distance_raw` | `TEXT` | `NULL` | `DISTANCE` | Route distance in statute miles (e.g. `'760.00'`). |

**Primary Key & Indexes:**
- `pk_bts_flight_staging`: `PRIMARY KEY (source_sha256, source_row_number)`.
- `ix_bts_staging_flight_date`: `ON bts_flight_staging (flight_date_raw)`.

---

## 4. Table: `public.silver_flights` (Silver)

Cleaned, strongly typed, validated, and normalized relational flight entity. Created by Migration 005.

| # | Column | Type | Nullable | Source & Transformation Logic | Description |
|---|---|---|---|---|---|
| 1 | `source_sha256` | `CHAR(64)` | `NOT NULL` | From Bronze | **PK Part 1**. Cryptographic provenance hash. |
| 2 | `source_row_number` | `BIGINT` | `NOT NULL` | From Bronze | **PK Part 2**. Source line number (`CHECK > 0`). |
| 3 | `ingestion_run_id` | `BIGINT` | `NOT NULL` | From Bronze | **FK** $\rightarrow$ `public.ingestion_runs(run_id)`. |
| 4 | `ingested_at` | `TIMESTAMPTZ` | `NOT NULL` | From Bronze | Bronze insertion timestamp. |
| 5 | `transformed_at` | `TIMESTAMPTZ` | `NOT NULL` | `NOW()` | Silver promotion timestamp. |
| 6 | `flight_date` | `DATE` | `NOT NULL` | Parsed from `flight_date_raw` | Scheduled departure calendar date. |
| 7 | `year` | `SMALLINT` | `NOT NULL` | Cast from `year_raw` | Calendar year (`CHECK >= 1987`). |
| 8 | `quarter` | `SMALLINT` | `NOT NULL` | Cast from `quarter_raw` | Calendar quarter (`CHECK 1..4`). |
| 9 | `month` | `SMALLINT` | `NOT NULL` | Cast from `month_raw` | Calendar month (`CHECK 1..12`). |
| 10 | `day_of_month` | `SMALLINT` | `NOT NULL` | Cast from `day_of_month_raw` | Calendar day of month (`CHECK 1..31`). |
| 11 | `day_of_week` | `SMALLINT` | `NOT NULL` | Cast from `day_of_week_raw` | ISO day of week (`1=Mon ... 7=Sun`). |
| 12 | `op_unique_carrier` | `TEXT` | `NOT NULL` | Trimmed `op_unique_carrier_raw` | Operating carrier IATA code (`CHECK length > 0`). |
| 13 | `origin_airport_id` | `BIGINT` | `NOT NULL` | Cast from `origin_airport_id_raw` | BTS numeric origin airport ID. |
| 14 | `origin` | `TEXT` | `NOT NULL` | Trimmed `origin_raw` | 3-letter IATA origin airport code. |
| 15 | `dest_airport_id` | `BIGINT` | `NOT NULL` | Cast from `dest_airport_id_raw` | BTS numeric destination airport ID. |
| 16 | `dest` | `TEXT` | `NOT NULL` | Trimmed `dest_raw` | 3-letter IATA destination airport code. |
| 17 | `crs_dep_time_raw` | `TEXT` | `NOT NULL` | Trimmed `crs_dep_time_raw` | Preserved raw departure time (e.g. `'0922'`, `'2400'`). |
| 18 | `crs_arr_time_raw` | `TEXT` | `NOT NULL` | Trimmed `crs_arr_time_raw` | Preserved raw arrival time (e.g. `'1145'`, `'2400'`). |
| 19 | `crs_dep_time_str` | `CHAR(4)` | `NOT NULL` | `LPAD(TRIM(crs_dep_time_raw), 4, '0')` | Standardized 4-digit HHMM departure string. |
| 20 | `crs_arr_time_str` | `CHAR(4)` | `NOT NULL` | `LPAD(TRIM(crs_arr_time_raw), 4, '0')` | Standardized 4-digit HHMM arrival string. |
| 21 | `crs_dep_time` | `TIME` | `NOT NULL` | Normalized clock time | Normalized `TIME` (00:00:00 to 23:59:59). `'2400'` maps to `'00:00:00'`. |
| 22 | `crs_arr_time` | `TIME` | `NOT NULL` | Normalized clock time | Normalized `TIME` (00:00:00 to 23:59:59). `'2400'` maps to `'00:00:00'`. |
| 23 | `crs_dep_timestamp` | `TIMESTAMP` | `NOT NULL` | Date + Time resolution | Scheduled departure local timestamp. For `'2400'`, rolls forward to midnight next day. |
| 24 | `crs_elapsed_time` | `DOUBLE PRECISION` | `NULL` | Cast from `crs_elapsed_time_raw` | Scheduled gate-to-gate elapsed time in minutes. Nullable. 5 negative values preserved. |
| 25 | `distance` | `DOUBLE PRECISION` | `NOT NULL` | Cast from `distance_raw` | Route distance in statute miles (`CHECK >= 0`). |
| 26 | `cancelled` | `BOOLEAN` | `NOT NULL` | Cast (`1.0` $\rightarrow$ `TRUE`) | Cancellation flag (`TRUE` = flight was cancelled). |
| 27 | `diverted` | `BOOLEAN` | `NOT NULL` | Cast (`1.0` $\rightarrow$ `TRUE`) | Diversion flag (`TRUE` = flight was diverted). |
| 28 | `dep_del15` | `SMALLINT` | `NULL` | Sanitized delay label | Binary delay label (1 = $\ge 15$ min, 0 = $< 15$ min). `NULL` for cancelled flights. |
| 29 | `is_training_eligible`| `BOOLEAN` | `NOT NULL` | Eligibility logic | `TRUE` if `dep_del15 IS NOT NULL AND cancelled = FALSE AND diverted = FALSE`. |

**Check Constraints:**
- `ck_silver_cancelled_dep_del15`: `CHECK (cancelled = FALSE OR dep_del15 IS NULL)` — prohibits delay labels on cancelled flights.
- `ck_silver_training_eligible`: `CHECK (is_training_eligible = (dep_del15 IS NOT NULL AND cancelled = FALSE AND diverted = FALSE))` — guarantees consistent training mask.
- `silver_flights_day_of_month_check`: `CHECK (day_of_month >= 1 AND day_of_month <= 31)`.
- `silver_flights_day_of_week_check`: `CHECK (day_of_week >= 1 AND day_of_week <= 7)`.
- `silver_flights_month_check`: `CHECK (month >= 1 AND month <= 12)`.
- `silver_flights_quarter_check`: `CHECK (quarter >= 1 AND quarter <= 4)`.
- `silver_flights_year_check`: `CHECK (year >= 1987)`.
- `silver_flights_distance_check`: `CHECK (distance >= 0)`.
- `silver_flights_origin_check`: `CHECK (length(origin) = 3)`.
- `silver_flights_dest_check`: `CHECK (length(dest) = 3)`.
- `silver_flights_op_unique_carrier_check`: `CHECK (length(op_unique_carrier) > 0)`.
- `silver_flights_source_row_number_check`: `CHECK (source_row_number > 0)`.

**Indexes:**
- `pk_silver_flights`: `PRIMARY KEY (source_sha256, source_row_number)`
- `ix_silver_flights_flight_date`: `ON silver_flights (flight_date)`
- `ix_silver_flights_dep_timestamp`: `ON silver_flights (crs_dep_timestamp)`
- `ix_silver_flights_training_eligible`: `ON silver_flights (flight_date) WHERE is_training_eligible = TRUE`
- `ix_silver_flights_origin_date`: `ON silver_flights (origin, flight_date)`
- `ix_silver_flights_carrier_date`: `ON silver_flights (op_unique_carrier, flight_date)`
- `ix_silver_flights_source_sha256`: `ON silver_flights (source_sha256)`

---

## 5. Table: `public.airport_reference`

Master reference table for airport dimensional attributes, geographic coordinates, and timezone offsets. Created by Migration 004.

| Column | Type | Nullable | Source Column | Description |
|---|---|---|---|---|
| `airport_seq_id` | `BIGINT` | `NOT NULL` | `AIRPORT_SEQ_ID` | **Primary Key**. Unique historical sequence identifier for airport version. |
| `airport_id` | `BIGINT` | `NOT NULL` | `AIRPORT_ID` | Stable BTS numeric identifier for an airport. |
| `airport_code` | `TEXT` | `NOT NULL` | `AIRPORT` | 3-letter IATA airport code (e.g. `'ATL'`, `'ORD'`). |
| `airport_name` | `TEXT` | `NULL` | | Full formal airport name (nullable). |
| `display_airport_name` | `TEXT` | `NULL` | `DISPLAY_AIRPORT_NAME` | Public display name (e.g. `'Atlanta, GA: Hartsfield-Jackson Atlanta International'`). |
| `latitude` | `DOUBLE PRECISION` | `NULL` | `LATITUDE` | Latitude in degrees (`CHECK BETWEEN -90 AND 90`). 1 record (ZZZ) is NULL. |
| `longitude` | `DOUBLE PRECISION` | `NULL` | `LONGITUDE` | Longitude in degrees (`CHECK BETWEEN -180 AND 180`). 1 record (ZZZ) is NULL. |
| `utc_local_time_variation` | `TEXT` | `NULL` | `UTC_LOCAL_TIME_VARIATION` | Time offset from UTC in minutes/hours (e.g. `'-0500'`). |
| `airport_start_date` | `DATE` | `NOT NULL` | `AIRPORT_START_DATE` | Date when this airport sequence version became active. |
| `airport_thru_date` | `DATE` | `NULL` | `AIRPORT_THRU_DATE` | Date when this airport sequence version retired (NULL if currently active). |
| `is_closed` | `BOOLEAN` | `NOT NULL` | `AIRPORT_IS_CLOSED` | `TRUE` if airport is decommissioned. |
| `is_latest` | `BOOLEAN` | `NOT NULL` | `AIRPORT_IS_LATEST` | `TRUE` if this record is the most recent snapshot for the `airport_id`. |
| `source_sha256` | `CHAR(64)` | `NOT NULL` | *(Derived)* | Provenance SHA-256 hash of `T_MASTER_CORD.csv`. |
| `source_row_number` | `BIGINT` | `NOT NULL` | *(Derived)* | CSV row index (`CHECK > 0`). |
| `ingestion_run_id` | `BIGINT` | `NOT NULL` | *(Derived)* | **FK** $\rightarrow$ `public.ingestion_runs(run_id)`. |
| `ingested_at` | `TIMESTAMPTZ` | `NOT NULL` | *(Derived)* | Ingestion timestamp. |

**Indexes & Constraints:**
- `airport_reference_pkey`: `PRIMARY KEY (airport_seq_id)`.
- `uq_airport_reference_source_row`: `UNIQUE (source_sha256, source_row_number)`.
- `ix_airport_reference_airport_id`: `ON airport_reference (airport_id)`.
- `ix_airport_reference_latest`: `ON airport_reference (airport_id) WHERE is_latest = TRUE`.
- `ix_airport_reference_source`: `ON airport_reference (source_sha256)`.

---

## 6. Table: `public.weather_observations`

Automated Surface Observing System (ASOS) raw surface weather observations from Iowa Environmental Mesonet. Created by Migration 002. Currently populated with January 2025 test dataset for 10 airports.

| Column | Type | Nullable | Source Header | Description |
|---|---|---|---|---|
| `station` | `TEXT` | `NOT NULL` | `station` | **PK Part 1**. 3- or 4-letter ICAO/IATA weather station identifier. |
| `observation_time` | `TIMESTAMP` | `NOT NULL` | `valid` | **PK Part 2**. Observation time in UTC (parsed without local offset distortion). |
| `source_sha256` | `CHAR(64)` | `NOT NULL` | *(Derived)* | SHA-256 fingerprint of the station CSV file. |
| `source_row_number` | `BIGINT` | `NOT NULL` | *(Derived)* | Line number in station CSV. |
| `ingestion_run_id` | `BIGINT` | `NOT NULL` | *(Derived)* | **FK** $\rightarrow$ `public.ingestion_runs(run_id)`. |
| `ingested_at` | `TIMESTAMPTZ` | `NOT NULL` | *(Derived)* | Ingestion timestamp. |
| `tmpf_raw` | `TEXT` | `NULL` | `tmpf` | Air temperature in degrees Fahrenheit. |
| `dwpf_raw` | `TEXT` | `NULL` | `dwpf` | Dew point temperature in degrees Fahrenheit. |
| `relh_raw` | `TEXT` | `NULL` | `relh` | Relative humidity percentage. |
| `drct_raw` | `TEXT` | `NULL` | `drct` | Wind direction in degrees from true North. |
| `sknt_raw` | `TEXT` | `NULL` | `sknt` | Wind speed in knots. |
| `p01i_raw` | `TEXT` | `NULL` | `p01i` | 1-hour precipitation in inches. |
| `alti_raw` | `TEXT` | `NULL` | `alti` | Altimeter pressure setting in inches of mercury. |
| `mslp_raw` | `TEXT` | `NULL` | `mslp` | Sea level pressure in hectopascals (hPa). |
| `vsby_raw` | `TEXT` | `NULL` | `vsby` | Visibility in statute miles. |
| `gust_raw` | `TEXT` | `NULL` | `gust` | Wind gust speed in knots. |
| `skyc1_raw` .. `skyc4_raw` | `TEXT` | `NULL` | `skyc1`..`skyc4` | Cloud coverage levels (e.g. `'CLR'`, `'FEW'`, `'SCT'`, `'BKN'`, `'OVC'`). |
| `skyl1_raw` .. `skyl4_raw` | `TEXT` | `NULL` | `skyl1`..`skyl4` | Cloud base heights in feet AGL. |
| `wxcodes_raw` | `TEXT` | `NULL` | `wxcodes` | METAR present weather codes (e.g. `'RA'`, `'SN'`, `'TS'`). |
| `metar_raw` | `TEXT` | `NULL` | `metar` | Full raw METAR report string. |

**Indexes:**
- `pk_weather_observations`: `PRIMARY KEY (station, observation_time)`.
- `ix_weather_observations_time`: `ON weather_observations (observation_time)`.
- `ix_weather_observations_source`: `ON weather_observations (source_sha256)`.

---

## 7. Table: `public.gold_features` (Gold V1)

> **Status:** **VERIFIED COMPLETE / LIVE** — Materialized in PostgreSQL via Migration 006 and `scripts/transform_gold_features.py`. Contains **6,879,484 rows** partitioned by `dataset_split`. Zero nulls, zero duplicate source rows. Population reconciled 100% against eligible Silver flights minus 1 excluded unphysical flight.

| # | Column | Type | Nullable | Role | Feature Source / Derivation |
|---|---|---|---|---|---|
| 1 | `source_sha256` | `CHAR(64)` | `NOT NULL` | Provenance Identifier | From `silver_flights`. **PK Part 1**. |
| 2 | `source_row_number` | `BIGINT` | `NOT NULL` | Provenance Identifier | From `silver_flights` (`CHECK > 0`). **PK Part 2**. |
| 3 | `flight_date` | `DATE` | `NOT NULL` | Provenance Identifier | Scheduled departure calendar date. |
| 4 | `dataset_split` | `TEXT` | `NOT NULL` | Partition Key | `'train'`, `'validation'`, or `'test'`. **PK Part 3**. |
| 5 | `dep_del15` | `SMALLINT` | `NOT NULL` | **Supervised Target** | 1 if departure delay $\ge 15$ min, else 0 (`CHECK (dep_del15 IN (0, 1))`). Strictly separated from predictors. |
| 6 | `year` | `SMALLINT` | `NOT NULL` | Temporal Predictor | Calendar year (`CHECK >= 1987`). |
| 7 | `month` | `SMALLINT` | `NOT NULL` | Temporal Predictor | Calendar month (`CHECK 1..12`). |
| 8 | `day_of_month` | `SMALLINT` | `NOT NULL` | Temporal Predictor | Calendar day of month (`CHECK 1..31`). |
| 9 | `day_of_week` | `SMALLINT` | `NOT NULL` | Temporal Predictor | ISO day of week (`CHECK 1..7`). |
| 10 | `quarter` | `SMALLINT` | `NOT NULL` | Temporal Predictor | Calendar quarter (`CHECK 1..4`). |
| 11 | `is_weekend` | `BOOLEAN` | `NOT NULL` | Temporal Predictor | `TRUE` if `day_of_week IN (6, 7)`. |
| 12 | `dep_hour` | `SMALLINT` | `NOT NULL` | Temporal Predictor | Scheduled departure hour (`CHECK 0..23`). BTS 2400 maps to 0. |
| 13 | `dep_minute` | `SMALLINT` | `NOT NULL` | Temporal Predictor | Scheduled departure minute (`CHECK 0..59`). |
| 14 | `sin_dep_time` | `DOUBLE PRECISION` | `NOT NULL` | Cyclical Predictor | $\sin(2\pi \cdot (\text{hour}\cdot 60 + \text{minute}) / 1440)$. |
| 15 | `cos_dep_time` | `DOUBLE PRECISION` | `NOT NULL` | Cyclical Predictor | $\cos(2\pi \cdot (\text{hour}\cdot 60 + \text{minute}) / 1440)$. |
| 16 | `sin_day_of_week` | `DOUBLE PRECISION` | `NOT NULL` | Cyclical Predictor | $\sin(2\pi \cdot (\text{day\_of\_week} - 1) / 7)$. |
| 17 | `cos_day_of_week` | `DOUBLE PRECISION` | `NOT NULL` | Cyclical Predictor | $\cos(2\pi \cdot (\text{day\_of\_week} - 1) / 7)$. |
| 18 | `sin_month` | `DOUBLE PRECISION` | `NOT NULL` | Cyclical Predictor | $\sin(2\pi \cdot (\text{month} - 1) / 12)$. |
| 19 | `cos_month` | `DOUBLE PRECISION` | `NOT NULL` | Cyclical Predictor | $\cos(2\pi \cdot (\text{month} - 1) / 12)$. |
| 20 | `op_unique_carrier` | `TEXT` | `NOT NULL` | Categorical Predictor | Operating carrier code (14 unique carriers). |
| 21 | `origin_airport_id` | `BIGINT` | `NOT NULL` | Categorical Predictor | BTS numeric origin airport ID. |
| 22 | `dest_airport_id` | `BIGINT` | `NOT NULL` | Categorical Predictor | BTS numeric destination airport ID. |
| 23 | `distance` | `DOUBLE PRECISION` | `NOT NULL` | Numeric Predictor | Route distance in statute miles (`CHECK >= 0`). |
| 24 | `crs_elapsed_time_clean` | `DOUBLE PRECISION` | `NOT NULL` | Numeric Predictor | Scheduled elapsed time in minutes (`CHECK > 0`). Populated directly from original positive Silver value. Nonpositive durations excluded. |
| 25 | `origin_latitude` | `DOUBLE PRECISION` | `NOT NULL` | Spatial Predictor | Point-in-time origin latitude from `airport_reference` (`CHECK -90..90`). |
| 26 | `origin_longitude` | `DOUBLE PRECISION` | `NOT NULL` | Spatial Predictor | Point-in-time origin longitude from `airport_reference` (`CHECK -180..180`). |
| 27 | `dest_latitude` | `DOUBLE PRECISION` | `NOT NULL` | Spatial Predictor | Point-in-time dest latitude from `airport_reference` (`CHECK -90..90`). |
| 28 | `dest_longitude` | `DOUBLE PRECISION` | `NOT NULL` | Spatial Predictor | Point-in-time dest longitude from `airport_reference` (`CHECK -180..180`). |

**Partitioning & Constraints:**
- Partitioned by `LIST (dataset_split)` into child tables: `gold_features_train`, `gold_features_val`, `gold_features_test`.
- Primary Key: `(source_sha256, source_row_number, dataset_split)`.
- No imputation flags or synthetic estimates. Invalid duration records are excluded prior to ingestion.

---

## 8. Weather Station Mapping Artifacts (Milestone M4)

Artifacts generated by `scripts/map_airport_weather_stations.py` to establish the station mapping dictionary for all 352 Gold origin airports.

### 8.1 File: `data/raw/weather/airport_inventory.csv`
Authoritative inventory of unique origin airports extracted from `public.gold_features`.
- **Rows:** 352
- **SHA-256:** `5b6812283b2b1bb22f73a41195ad24c2e10b489e83bd0e0c58c8a1750c4b91b3`
- **Columns:**
  - `origin_airport_id` (`BIGINT`, PK): BTS unique origin airport ID.
  - `latitude` (`DOUBLE PRECISION`): Origin airport latitude in decimal degrees.
  - `longitude` (`DOUBLE PRECISION`): Origin airport longitude in decimal degrees.
  - `flight_count` (`BIGINT`): Total 2025 flights originating from this airport (Sum = 6,879,484).
  - `active_days` (`INTEGER`): Distinct flight dates observed in 2025.

### 8.2 File: `data/processed/weather/airport_station_mapping.csv`
Authoritative 1-to-1 mapping linking each origin airport to its selected primary weather station.
- **Rows:** 352 (Zero unmapped airports, zero duplicates).
- **Columns:**
  - `origin_airport_id` (`BIGINT`, PK): BTS unique origin airport ID.
  - `airport_code` (`TEXT`): Point-in-time IATA airport code (e.g. `'ATL'`, `'DFW'`).
  - `airport_name` (`TEXT`): Airport facility name.
  - `airport_latitude`, `airport_longitude` (`DOUBLE PRECISION`): Airport reference coordinates.
  - `station_id` (`TEXT`): Selected IEM ASOS station identifier (e.g. `'ATL'`, `'PANC'`, `'IWA'`).
  - `station_name` (`TEXT`): Station description from IEM catalogue.
  - `station_latitude`, `station_longitude` (`DOUBLE PRECISION`): Weather station coordinates.
  - `elevation_m` (`DOUBLE PRECISION`, Nullable): Station elevation above sea level in meters.
  - `distance_km` (`DOUBLE PRECISION`): Great-circle distance between airport and weather station (Max: 3.42 km).
  - `match_type` (`TEXT`): Categorical match classification (`EXACT_IATA_ON_AIRPORT`, `ICAO_PREFIX_ON_AIRPORT`, `AIRPORT_FACILITY_COORDINATE_MATCH`).
  - `selection_reason` (`TEXT`): Audit trail describing why the station was chosen.
  - `network` (`TEXT`): IEM network identifier (e.g. `'TX_ASOS'`, `'AK_ASOS'`).
  - `archive_begin`, `archive_end` (`TEXT`): Provenance archive operational window from catalogue.

### 8.3 File: `data/processed/weather/airport_station_candidates.csv`
Ranked candidate evaluation provenance capturing up to 5 nearest weather stations per airport.
- **Rows:** 1,651 ranked candidate entries.
- **Key Columns:** `airport_id`, `airport_code`, `candidate_rank` (1..5), `station_id`, `distance_km`, `is_exact_code`, `is_icao_match`, `archive_covers_2025`, `is_selected`.

### 8.4 File: `data/processed/weather/airport_station_mapping_summary.json`
Machine-readable audit manifest summarizing mapping counts, distance distribution, catalogue provenance hash, and acceptance criteria verification.

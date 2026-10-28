# FlightOps ML — Architecture Decision Records (ADRs)

This document tracks all key technical, architectural, and data modeling decisions made in the FlightOps ML project.

---

## Decision Log Summary

| ADR ID | Date | Status | Title | Summary |
|---|---|---|---|---|
| [ADR-001](#adr-001-raw-ingestion-architecture-bronze-staging--audit-ledger) | 2026-10-15 | **Accepted** | Raw Ingestion Architecture (Bronze Staging & Audit Ledger) | Untyped text staging, streaming COPY, SHA-256 provenance, and audit tracking. |
| [ADR-002](#adr-002-silver-normalization--strict-typing) | 2026-10-16 | **Accepted** | Silver Normalization & Strict Typing | Strong relational typing, explicit constraints, and 100% Bronze reconciliation. |
| [ADR-003](#adr-003-target-definition--cancellation-policy) | 2026-10-16 | **Accepted** | Target Definition & Cancellation Policy | Binary `dep_del15`; cancelled flights strictly NULL; training mask enforced. |
| [ADR-004](#adr-004-handling-time-edge-case-2400) | 2026-10-16 | **Accepted** | Handling Time Edge Case `2400` | Clock time `00:00:00`, timestamp midnight next day, `flight_date` preserved. |
| [ADR-005](#adr-005-treatment-of-negative-scheduled-elapsed-time-records) | 2026-10-16 | **Accepted** | Treatment of Negative Scheduled Elapsed-Time Records | Preserve in Silver; address via route median imputation in Gold tier. |
| [ADR-006](#adr-006-chronological-trainvalidationtest-split-boundaries) | 2026-10-19 | **Accepted** | Chronological Train / Validation / Test Split Boundaries | 8-month Train (Jan–Aug), 2-month Val (Sep–Oct), 2-month Test (Nov–Dec). |
| [ADR-007](#adr-007-two-hour-scheduled-departure-prediction-cutoff) | 2026-10-19 | **Proposed** | Two-Hour Scheduled Departure Prediction Cutoff | Operational cutoff at `CRS_DEP_TIMESTAMP - 2 hours`; zero post-cutoff features. |
| [ADR-008](#adr-008-airport-reference-join-strategy-point-in-time-by-airport_id) | 2026-10-20 | **Accepted** | Airport Reference Join Strategy (Point-in-Time by `airport_id`) | Join by `origin_airport_id` + date window to prevent dropping renamed codes (PBI $\rightarrow$ DJT). |
| [ADR-009](#adr-009-weather-data-integration-strategy) | 2026-10-20 | **Proposed** | Weather Data Integration Strategy (Point-in-Time Correctness) | Evaluate 2-hour lagged ASOS observations vs. NOAA HRRR/RAP numerical forecasts. |
| [ADR-010](#adr-010-gold-feature-storage-strategy) | 2026-10-20 | **Accepted** | Gold Feature Storage Strategy (Materialized Table vs. View) | Declarative list partitioning on `dataset_split`. |
| [ADR-011](#adr-011-logistic-regression-baseline-specification-sparse-preprocessing--evaluation-protocol) | 2026-10-20 | **Proposed** | Logistic Regression Baseline Specification | Sparse CSR preprocessing pipeline, balanced class weights, strict split isolation. |
| [ADR-012](#adr-012-weather-station-mapping-methodology--evidence-based-station-selection-for-352-origin-airports) | 2026-10-21 | **Accepted** | Weather Station Mapping Methodology & Station Selection | Evidence-based mapping for 352 origin airports using Haversine distance, IATA/ICAO codes, and archive verification. |

---

## ADR-001: Raw Ingestion Architecture (Bronze Staging & Audit Ledger)

- **Date:** 2026-10-15
- **Status:** **Accepted**
- **Context:** BTS On-Time Reporting archives contain millions of rows per month. Ingestion must be fast, resilient to format discrepancies, idempotent, and completely auditable.
- **Decision:**
  1. Store all incoming CSV columns as untyped `TEXT` in `public.bts_flight_staging` (Bronze tier).
  2. Implement an audit table `public.ingestion_runs` tracking `run_id`, `source_name`, `source_sha256`, `records_seen`, `records_inserted`, and `records_skipped`.
  3. Use PostgreSQL `COPY FROM STDIN` streaming via a temporary staging table to achieve high ingestion throughput (>50,000 rows/sec).
  4. Deduplicate using a composite primary key `(source_sha256, source_row_number)` with `ON CONFLICT DO NOTHING`.
- **Consequences:** All 12 monthly archives for 2025 (7,001,619 rows) ingested without type coercion failures or unlogged data loss.

---

## ADR-002: Silver Normalization & Strict Typing

- **Date:** 2026-10-16
- **Status:** **Accepted**
- **Context:** Raw text records in Bronze cannot be reliably queried or modeled. Downstream pipelines need strongly typed, validated dates, times, carriers, and numeric values with guaranteed invariants.
- **Decision:**
  1. Create `public.silver_flights` with native PostgreSQL types: `DATE`, `TIME`, `TIMESTAMP`, `SMALLINT`, `BIGINT`, `DOUBLE PRECISION`, `BOOLEAN`.
  2. Implement idempotent batch transformation per monthly archive via `scripts/transform_silver_flights.py`.
  3. Pre-validate each Bronze partition before transformation; abort immediately if unexpected date formats, invalid carrier codes, or airport string anomalies are found.
  4. Enforce domain invariants via check constraints (e.g., month 1..12, valid airport string length, positive distance).
- **Consequences:** 100% 1-to-1 parity between Bronze and Silver verified across all 7,001,619 records.

---

## ADR-003: Target Definition & Cancellation Policy

- **Date:** 2026-10-16
- **Status:** **Accepted**
- **Context:** In airline operations, a flight cancellation is an operational disruption, but not an observed departure delay. Imputing missing delay labels on cancelled flights to `0` would falsely label them as on-time departures; imputing to `1` would corrupt delay prediction.
- **Decision:**
  1. Target is `dep_del15` ($\ge 15$ minutes late = 1, $< 15$ minutes = 0).
  2. All cancelled flights must have `dep_del15 = NULL`, enforced at the database level by check constraint `ck_silver_cancelled_dep_del15`.
  3. Create an explicit training eligibility boolean column:
     `is_training_eligible = (dep_del15 IS NOT NULL AND cancelled = FALSE AND diverted = FALSE)`.
  4. Retain cancellations and diversions in Silver for operational analysis, but filter them out during supervised delay training.
- **Consequences:** Zero cancelled flights have delay labels. Supervised training is restricted to the 6,879,485 flights with observed departure outcomes.

---

## ADR-004: Handling Time Edge Case `2400`

- **Date:** 2026-10-16
- **Status:** **Accepted**
- **Context:** BTS uses `2400` to represent scheduled midnight departures and arrivals (occurring in 3 flights in 2025). Standard Python `datetime.time` and SQL `TIME` reject hour 24.
- **Decision:**
  1. Preserve raw strings in `crs_dep_time_raw`, `crs_arr_time_raw`, `crs_dep_time_str`, and `crs_arr_time_str`.
  2. For clock time columns (`crs_dep_time`, `crs_arr_time`), map `'2400'` to `TIME '00:00:00'`.
  3. For scheduled timestamp (`crs_dep_timestamp`), roll `'2400'` forward to midnight of the following calendar day (`flight_date + INTERVAL '1 day' + TIME '00:00:00'`).
  4. The operational `flight_date` itself is **never silently incremented**.
- **Consequences:** Preserves operational reporting dates while avoiding runtime exceptions in Python and PostgreSQL.

---

## ADR-005: Treatment of Missing and Nonpositive Scheduled Elapsed-Time Records (Exclusion Policy)

- **Date:** 2026-10-16 (Revised 2026-10-20)
- **Status:** **Accepted** (Supersedes previous route-median imputation proposal)
- **Context:**
  - An audit of the full 2025 Silver dataset (`silver_flights`, 7,001,619 total rows) revealed exactly 7 records with missing or nonpositive `crs_elapsed_time`:
    - 2 records with `crs_elapsed_time IS NULL` (both Hawaiian Airlines cancelled flights, `is_training_eligible = FALSE`).
    - 4 records with `crs_elapsed_time < 0` on Southwest Airlines (3 diverted flights and 1 cancelled flight, all `is_training_eligible = FALSE`).
    - 1 record with `crs_elapsed_time < 0`: 2025-11-19 SkyWest (`OO`) DTW $\rightarrow$ LEX (`crs_elapsed_time = -60.0`, completed flight, `is_training_eligible = TRUE`, test split).
  - Across the 6,879,485 training-eligible Silver flights, there are zero missing or nonpositive duration records in the training split, zero in the validation split, and exactly one in the test split.
  - The previously proposed route-median / global-median imputation framework was disproportionate and unnecessary for a single unphysical test record.
- **Decision:**
  1. **Preserve Anomalies in Silver:** Retain all 7 anomalous records in Silver to maintain raw audit fidelity and complete source provenance.
  2. **Explicit Gold Exclusion:** Exclude eligible Silver flights with missing or nonpositive duration from the Gold feature table before transformation:
     `WHERE s.is_training_eligible = TRUE AND s.crs_elapsed_time IS NOT NULL AND s.crs_elapsed_time > 0`.
  3. **Direct Population:** Populate `crs_elapsed_time_clean` directly from the original positive Silver value without alteration, enforced by `CHECK (crs_elapsed_time_clean > 0)`.
  4. **Remove Imputation Framework:** Remove all median calculation code, `ImputationStats`, imputation metadata JSON files, and the `crs_elapsed_time_imputed` column.
  5. **Exact Row Reconciliation:**
     - **Train:** 4,592,366 eligible Silver − 0 excluded = **4,592,366** Gold rows.
     - **Validation:** 1,159,899 eligible Silver − 0 excluded = **1,159,899** Gold rows.
     - **Test:** 1,127,220 eligible Silver − 1 excluded = **1,127,219** Gold rows.
     - **Total:** 6,879,485 eligible Silver − 1 excluded = **6,879,484** Gold rows.
- **Consequences:** Eliminates pipeline complexity, avoids inventing synthetic durations, and guarantees zero corrupted or unphysical duration records enter model training or evaluation.

---

## ADR-006: Chronological Train/Validation/Test Split Boundaries

- **Date:** 2026-10-19
- **Status:** **Accepted**
- **Context:** Random train/test splits cause severe temporal leakage in flight operations due to seasonal trends, operational autocorrelation, and weather patterns.
- **Decision:**
  1. Enforce a strict chronological split across calendar year 2025:
     - **Train:** January 1, 2025 through August 31, 2025 (243 days, 4,592,366 eligible flights, 22.19% delay rate).
     - **Validation:** September 1, 2025 through October 31, 2025 (61 days, 1,159,899 eligible flights, 18.08% delay rate).
     - **Test:** November 1, 2025 through December 31, 2025 (61 days, 1,127,220 eligible flights, 23.40% delay rate).
  2. The Test partition is strictly isolated: never used for model selection, threshold tuning, or feature scaling.
- **Consequences:** Guarantees realistic forward-in-time evaluation reflecting production deployment conditions.

---

## ADR-007: Two-Hour Scheduled Departure Prediction Cutoff

- **Date:** 2026-10-19
- **Status:** **Proposed**
- **Context:** Predictions must be delivered with sufficient lead time for airline operations to act (e.g., reassign gates, notify passengers, reschedule crew).
- **Proposal:**
  - Define the prediction cutoff as exactly 2 hours prior to scheduled departure:
    $$T_{\text{cutoff}} = \text{CRS\_DEP\_TIMESTAMP} - 2\text{ hours}$$
  - Disallow any feature whose availability cannot be demonstrated at $T_{\text{cutoff}}$.
- **Status:** **Proposed** — Pending validation of weather and inbound aircraft features at $T_{\text{cutoff}}$.

---

## ADR-008: Airport Reference Join Strategy (Point-in-Time by `airport_id`)

- **Date:** 2026-10-20
- **Status:** **Accepted**
- **Context:** Joining `silver_flights` to `airport_reference` by `airport_code` with `is_latest = TRUE` fails for Palm Beach International (`PBI`), which was renamed to `DJT` in snapshot updates starting August 2026.
- **Decision:**
  - Join by `origin_airport_id` / `dest_airport_id` with point-in-time date validity:
    `flight_date BETWEEN airport_start_date AND COALESCE(airport_thru_date, '9999-12-31')`.
- **Consequences:** Resolves 100% (352/352) of origin and destination airports without missing values or synthetic key failures.

---

## ADR-009: Weather Data Integration Strategy

- **Date:** 2026-10-20
- **Status:** **Proposed (Under Investigation)**
- **Context:** Surface weather observations (ASOS/METAR) occur in real time and are recorded in UTC. Joining observations at actual departure time leaks future information; joining observations at scheduled departure time violates the 2-hour cutoff.
- **Proposal:**
  1. For ASOS surface observations: join the latest observation occurring on or before $T_{\text{cutoff}}$ (i.e. at least 2 hours before scheduled departure in UTC).
  2. For weather forecasts: investigate historical NOAA HRRR (High-Resolution Rapid Refresh) or RAP numerical forecasts valid for $T_{\text{dep}}$ issued prior to $T_{\text{cutoff}}$.
- **Status:** Under investigation. Prototype ASOS loader verified for 10 airports in January 2025.

---

## ADR-010: Gold Feature Storage Strategy

- **Date:** 2026-10-20
- **Status:** **Accepted**
- **Context:** The Gold tier contains 6.88M eligible records. We must decide how to store Gold features to ensure fast partition access without risking cross-split leakage.
- **Decision:**
  - Adopt declarative PostgreSQL list partitioning on `dataset_split`:
    - `public.gold_features` partitioned by `LIST (dataset_split)`.
    - Child tables: `gold_features_train`, `gold_features_val`, `gold_features_test`.
    - Primary key constraint: `PRIMARY KEY (source_sha256, source_row_number, dataset_split)`.
  - Downstream ML pipelines can export partitions to compressed Parquet (`data/processed/gold/{split}.parquet`) once Gold V1 passes review and live database reconciliation.
- **Consequences:** Ensures partition pruning during SQL queries, physically segregates test data on disk, and provides clean support for high-throughput columnar ML loading.

---

## ADR-011: Logistic Regression Baseline Specification (Sparse Preprocessing & Evaluation Protocol)

- **Date:** 2026-10-20
- **Status:** **Proposed** (Awaiting approval before implementation)
- **Context:**
  - Milestone 4 Step 2 establishes the no-weather linear benchmark on `public.gold_features`.
  - The training split contains 4,592,366 rows. A dense one-hot encoding of categorical variables (14 carriers, 349 origin airports, 349 destination airports) would produce $>700$ columns, requiring $\approx 27\text{ GB}$ of RAM for a single dense array and causing severe memory exhaustion.
  - Three origin airports in the holdout test set (11577, 14098, 15628) do not appear in the training split due to seasonality.
  - Class distribution is imbalanced: 22.19% delayed in train, 18.08% delayed in validation ($\approx 3.5:1$ negative-to-positive ratio).
- **Decision:**
  1. **Sparse Feature Pipeline:**
     - Preprocess via Scikit-Learn `ColumnTransformer` yielding a `scipy.sparse.csr_matrix`.
     - Categorical variables (`op_unique_carrier`, `origin_airport_id`, `dest_airport_id`): `OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)`.
     - Continuous numeric variables (`distance`, `crs_elapsed_time_clean`, coordinates): `StandardScaler()` fitted strictly on training data.
     - Cyclical terms (`sin_dep_time`, `cos_dep_time`, etc.): pass-through (already bounded $[-1, 1]$).
     - **Memory Footprint:** $\approx 862\text{ MB}$ (float32 CSR) vs. $26.9\text{ GB}$ (dense float64) — a 95%+ memory reduction.
  2. **Model Estimator:**
     - `LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced", random_state=42)`.
     - Fits natively on sparse CSR matrices and converges efficiently with L2 regularization.
  3. **Baseline Comparison & Metrics Protocol:**
     - **Majority Class Naive Baseline:** Always predict $\hat{y} = 0$ (validation accuracy = 81.92%, Precision = 0.0, Recall = 0.0, F1 = 0.0, ROC-AUC = 0.5000, PR-AUC = 0.1808, Brier score = 0.1808).
     - **Primary Model Metrics (Validation Split):**
       - Ranking: ROC-AUC.
       - Imbalanced Precision / Calibration: PR-AUC (Average Precision), Brier Score.
       - Operational Decision Quality: Precision, Recall, F1 at threshold $\tau = 0.5$ and optimal F1 threshold.
  4. **Strict Split Discipline:**
     - Preprocessor and model fit strictly on `gold_features_train`.
     - Model evaluation and decision threshold tuning performed strictly on `gold_features_val`.
     - `gold_features_test` is strictly isolated and never accessed during baseline modeling.
- **Consequences:** Eliminates memory exhaustion risks, safely handles seasonal unseen categories, and establishes a defensible, reproducible benchmark for subsequent gradient boosting models.

---

## ADR-012: Weather Station Mapping Methodology & Evidence-Based Station Selection for 352 Origin Airports

- **Date:** 2026-10-21
- **Status:** **Accepted**
- **Context:**
  - Milestone M4 expands weather feature coverage from the initial 10 test stations to all 352 unique origin airports represented in `public.gold_features`.
  - Input catalogue: Iowa Environmental Mesonet (IEM) ASOS/AWOS station catalogue (`data/raw/weather/weather-station-catalogue.csv`, 6,047 stations, SHA-256: `2c8f8f8e28f51123e8aa2e45077601df0d32eaa5f1568e21ef0b317220f59da3`).
  - Airport inventory: Extracted directly from `public.gold_features` via `database_connection()` (`data/raw/weather/airport_inventory.csv`, 352 unique airports, 6,879,484 flights, SHA-256: `5b6812283b2b1bb22f73a41195ad24c2e10b489e83bd0e0c58c8a1750c4b91b3`).
- **Decision:**
  1. **Evidence-Based Matching & Candidate Ranking:**
     - Compute great-circle distance between airport coordinates and catalogue stations using the Haversine formula ($R = 6,371.0088\text{ km}$).
     - For every airport, rank the top 5 nearest candidates ascending by distance and verify against multiple independent evidence sources:
       - Exact 3-letter IATA code match (`stid == airport_code`).
       - Territorial 4-letter ICAO code match (`stid` with standard prefix `K`, `P`, `PA`, `PH`, `TJ`, `TI`, `PG`, `NS`).
       - Colocated airfield coordinate and facility alignment (FAA LID or local municipal facility name).
       - Active 2025 archive verification (`begints <= '2025-01-01'` and `endts == ''` or $\ge \text{'2025-12-31'}$).
  2. **Hierarchical Match Types:**
     - `EXACT_IATA_ON_AIRPORT`: 310 airports (88.07%). Exact identifier match on airport field ($d \le 3.42\text{ km}$, median $0.32\text{ km}$).
     - `ICAO_PREFIX_ON_AIRPORT`: 8 airports (2.27%). Standard territorial ICAO code on airport field (e.g. Alaska `PADK`, `PADQ`, `PAKN`, `PANC`, `PAWD`; Pacific `NSTU` for Pago Pago; `PGSN` for Saipan).
     - `AIRPORT_FACILITY_COORDINATE_MATCH`: 34 airports (9.66%). Colocated airfield ASOS station verified on runway grounds ($d \le 1.58\text{ km}$, median $0.21\text{ km}$) where FAA LID or historic facility code differs from BTS IATA (e.g. `AZA` $\rightarrow$ `IWA`, `CLD` $\rightarrow$ `CRQ`, `FCA` $\rightarrow$ `GPI`, `HHH` $\rightarrow$ `HXD`, `USA` $\rightarrow$ `JQF`, `MQT` $\rightarrow$ `SAW`, `SCE` $\rightarrow$ `UNV`, `YUM` $\rightarrow$ `NYL`, `GUF` $\rightarrow$ `JKA`, `PBI` $\rightarrow$ `DJT`).
     - `NEARBY_FALLBACK_VICINITY`: 0 airports. No airport required off-airport regional fallback ($d > 5.0\text{ km}$).
     - `FLAGGED_FOR_MANUAL_REVIEW`: 0 airports. Zero airports lacked a defensible on-field station.
  3. **Coverage & Proximity Metrics:**
     - **Total Origin Airports Mapped:** 352 / 352 (100.0%).
     - **Distance Summary:** Minimum: 0.000 km, Median: 0.342 km, Mean: 0.509 km, Maximum: 3.417 km (100% within 3.5 km).
     - **Active 2025 Archive Status:** 352 / 352 (100.0%) stations have continuous active archives covering the entire 2025 calendar year.
     - **Station Uniqueness:** 352 unique weather stations mapped to 352 origin airports (zero duplicate station assignments).
- **Critical Distinction:**
  - Archive metadata in the catalogue proves that the station was commissioned, operational, and archived by IEM throughout 2025.
  - However, **catalogue archive dates do not guarantee zero missing hourly observations or complete sensor coverage**.
- **Next Planned Step:**
  - Execute a pre-download data availability audit to verify actual hourly observation counts and null rates for each of the 352 stations across the period January 1 through December 31, 2025, before initiating bulk observation ingestion.

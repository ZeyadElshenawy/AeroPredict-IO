# FlightOps ML — Data Quality, Edge Cases & Leakage Audit

> **Audit Status:** Verified against active codebase and PostgreSQL container on October 20, 2026.
> **Target Scope:** BTS On-Time Performance (2025), Airport Reference, and Surface Weather.

---

## 1. Verified Integrity Checks & Baseline Audit

The following table summarizes the verified database-level assertions performed across the Bronze and Silver layers:

| Assertion / Rule | Expected | Verified Result | Evidence / SQL Check |
|---|---|---|---|
| **Bronze Total Rows** | 7,001,619 | **7,001,619** | `SELECT COUNT(*) FROM public.bts_flight_staging` |
| **Silver Total Rows** | 7,001,619 | **7,001,619** | `SELECT COUNT(*) FROM public.silver_flights` |
| **Bronze-to-Silver Parity** | 100.0% | **100.0%** | Zero skipped or dropped records; 1-to-1 match per monthly partition |
| **Primary Key Uniqueness** | Zero duplicate keys | **Zero duplicates** | `COUNT(DISTINCT (source_sha256, source_row_number)) = 7,001,619` |
| **Cancelled Delay Labels** | Exactly 0 with label | **0** | `COUNT(*) FILTER (WHERE cancelled = TRUE AND dep_del15 IS NOT NULL) = 0` |
| **Training Eligibility Rule** | Zero violations | **0 violations** | Enforced by check constraint `ck_silver_training_eligible` |
| **Negative Elapsed Times** | Exactly 5 rows | **5 rows** | `COUNT(*) FILTER (WHERE crs_elapsed_time < 0) = 5` |
| **Operating Carriers** | 14 carriers | **14 carriers** | `COUNT(DISTINCT op_unique_carrier) = 14` |
| **Origin / Destination Airports** | 352 distinct | **352 origins, 352 dests**| `COUNT(DISTINCT origin) = 352`, `COUNT(DISTINCT dest) = 352` |
| **Date Coverage** | 365 calendar days | **365 days** | `2025-01-01` to `2025-12-31` |
| **Eligible Flights** | 6,879,485 | **6,879,485** | Non-cancelled, non-diverted with observed departure label |
| **Eligible Delay Rate** | ~21.7% | **21.69%** | 1,492,443 delayed flights / 6,879,485 total eligible |

---

## 2. Known Data Anomalies & Normalization Policies

### 2.1 The Five Negative Elapsed-Time Flights

In the raw BTS dataset for 2025, five flights possess negative values for scheduled elapsed time (`crs_elapsed_time_raw`). Rather than discarding these records or silently imputing them during Silver transformation, they are preserved in Silver for auditability.

#### Inspected Records

```
 source_sha256: 6235db7c5ca7f3fe678b6be805eab3ea1c2b78b2129e554721c2659501348f0e (2025-11.zip)
 ├── Row 374213: 2025-11-20 | WN | HOU -> DAL | Dep: 1635 | Arr: 1510 | Elapsed:  -85 | Diverted=T | Eligible=FALSE
 ├── Row 318592: 2025-11-17 | WN | LAS -> BUR | Dep: 1724 | Arr: 1545 | Elapsed:  -99 | Diverted=T | Eligible=FALSE
 ├── Row 318593: 2025-11-17 | WN | LAS -> BUR | Dep: 1802 | Arr: 1655 | Elapsed:  -67 | Diverted=T | Eligible=FALSE
 └── Row 349025: 2025-11-19 | OO | DTW -> LEX | Dep: 1053 | Arr: 0953 | Elapsed:  -60 | Diverted=F | Eligible=TRUE [!]

 source_sha256: 3b28769d529056e661fe5efa490b0021d2a1b377b91b22d0fae6d3b6eb7285ca (2025-12.zip)
 └── Row 392013: 2025-12-21 | WN | HOU -> HRL | Dep: 0056 | Arr: 2355 | Elapsed:  -61 | Cancelled=T| Eligible=FALSE
```

#### Diagnostic & Modeling Implications
1. **Four Ineligible Rows:** Rows 374213, 318592, 318593, and 392013 are either diverted or cancelled. They are automatically flagged `is_training_eligible = FALSE` and will not enter supervised delay training.
2. **One Test-Eligible Row:** Row 349025 (SkyWest / `OO` from Detroit `DTW` to Lexington `LEX` on `2025-11-19`) has `cancelled = FALSE`, `diverted = FALSE`, and `dep_del15 = 0`. Both DTW and LEX are in the US Eastern Time Zone. The scheduled departure was 10:53 AM and arrival was 09:53 AM, reflecting a 1-hour clerical schedule reversal in the carrier's published CRS timetable.
3. **Gold Policy:** In `gold_features`, `crs_elapsed_time` must not be fed raw into models that assume strictly positive durations. A feature `crs_elapsed_time_clean` will replace values $\le 0$ with the route median elapsed time (for DTW $\rightarrow$ LEX: ~70 minutes), accompanied by a boolean indicator `crs_elapsed_time_anomalous`.

---

### 2.2 Time Representation & The `2400` Midnight Edge Case

BTS reports times in military string format `HHMM`. In 2025, three records use `'2400'` (one departure and two arrivals):
- Python `datetime.time` strictly rejects hour 24 (`0 <= hour < 24`).
- SQL `TIME WITHOUT TIME ZONE` in PostgreSQL also rejects `'24:00:00'` in many standard casts.

#### Handled Convention:
- `crs_dep_time_raw` & `crs_arr_time_raw`: Preserved verbatim (`'2400'`).
- `crs_dep_time_str` & `crs_arr_time_str`: Preserved as standardized 4-digit strings (`'2400'`).
- `crs_dep_time` & `crs_arr_time`: Normalized to standard clock time `TIME '00:00:00'`.
- `crs_dep_timestamp`: When departure time is `'2400'`, the timestamp is constructed as midnight of the **following calendar day** (`(flight_date + INTERVAL '1 day' + TIME '00:00:00')::TIMESTAMP`).
- **Critical Invariant:** The operational `flight_date` itself is **never silently incremented**. A flight scheduled on July 24 at 24:00 remains operationally assigned to `flight_date = 2025-07-24`.

---

### 2.3 Cancellations, Diversions & Target Integrity

In airline operations, a cancelled flight is **not** an on-time flight:
- **Flawed Anti-Pattern:** Converting missing delay labels for cancelled flights to `0` (which falsely labels cancellations as successful on-time departures) or to `1` (which conflates cancellation with delay).
- **Implemented Contract:**
  1. All 98,719 cancelled flights in 2025 have `dep_del15 = NULL`, guaranteed by database constraint `ck_silver_cancelled_dep_del15`.
  2. All 23,415 diverted flights and 98,719 cancelled flights have `is_training_eligible = FALSE`.
  3. The supervised target `dep_del15` is strictly defined on observed departures.

---

### 2.4 Master Airport Reference Join Subtlety (Code Renaming)

When matching `public.silver_flights` against `public.airport_reference`:
- Matching naively on `origin = airport_code WHERE is_latest = TRUE` yields only **351 matches out of 352**.
- **The Missing Airport:** Palm Beach International (`PBI`, BTS numeric ID `14027`).
- **Root Cause:** In the master coordinate file (`T_MASTER_CORD.csv`), `airport_id = 14027` has three historical records:
  - `1402701`: `PBI` active 1950-01-01 to 2011-06-30 (`is_latest = FALSE`).
  - `1402702`: `PBI` active 2011-07-01 to 2026-07-31 (`is_latest = FALSE`).
  - `1402703`: `DJT` active 2026-08-01 onwards (`is_latest = TRUE`).
- **Required Join Pattern:** Downstream feature queries must join on `airport_id` with point-in-time validity:
  ```sql
  LEFT JOIN public.airport_reference ar
    ON s.origin_airport_id = ar.airport_id
   AND s.flight_date >= ar.airport_start_date
   AND (ar.airport_thru_date IS NULL OR s.flight_date <= ar.airport_thru_date)
  ```
  Joining on `origin_airport_id` with this date window resolves **100% (352/352)** of origin and destination airports.

---

### 2.5 Scheduled Elapsed-Time Anomalies and Explicit Exclusion

An exhaustive database audit of the 2025 Silver table (`silver_flights`, 7,001,619 total rows) revealed exactly 7 records with `crs_elapsed_time IS NULL OR crs_elapsed_time <= 0`:

| Flight Date | Carrier | Origin $\rightarrow$ Dest | `crs_elapsed_time` | Cancelled | Diverted | `dep_del15` | `is_training_eligible` | Split |
|---|---|---|---:|:---:|:---:|:---:|:---:|:---:|
| 2025-06-26 | `HA` | HNL $\rightarrow$ OGG | `NULL` | `TRUE` | `FALSE` | `NULL` | `FALSE` | Train |
| 2025-06-27 | `HA` | LIH $\rightarrow$ HNL | `NULL` | `TRUE` | `FALSE` | `NULL` | `FALSE` | Train |
| 2025-11-17 | `WN` | LAS $\rightarrow$ BUR | -99.0 | `FALSE` | `TRUE` | 0 | `FALSE` | Test |
| 2025-11-17 | `WN` | LAS $\rightarrow$ BUR | -67.0 | `FALSE` | `TRUE` | 0 | `FALSE` | Test |
| 2025-11-19 | `OO` | DTW $\rightarrow$ LEX | -60.0 | `FALSE` | `FALSE` | 0 | **`TRUE`** | Test |
| 2025-11-20 | `WN` | HOU $\rightarrow$ DAL | -85.0 | `FALSE` | `TRUE` | 0 | `FALSE` | Test |
| 2025-12-21 | `WN` | HOU $\rightarrow$ SAT | -61.0 | `TRUE` | `FALSE` | `NULL` | `FALSE` | Test |

#### Audit Reconciliation:
- **Broad Silver Audit vs. Eligible Test Flight:** The earlier audit observed 5 negative records in Silver. However, 3 were diverted flights and 1 was cancelled. Under Silver training eligibility rules (`cancelled = FALSE AND diverted = FALSE AND dep_del15 IS NOT NULL`), those 4 are **already ineligible** (`is_training_eligible = FALSE`).
- Therefore, among the **6,879,485 training-eligible Silver flights**, exactly **1 flight** (SkyWest OO on 2025-11-19, test split) has a negative scheduled duration, and zero have null durations.
- **Policy:** The 7 records are preserved in Silver for audit fidelity. In Gold V1, rather than synthesizing durations via an imputation framework, eligible flights are explicitly filtered:
  ```sql
  WHERE s.is_training_eligible = TRUE
    AND s.crs_elapsed_time IS NOT NULL
    AND s.crs_elapsed_time > 0
  ```
- **Reconciled Counts:**
  - Train: 4,592,366 eligible − 0 excluded = **4,592,366** Gold rows.
  - Validation: 1,159,899 eligible − 0 excluded = **1,159,899** Gold rows.
  - Test: 1,127,220 eligible − 1 excluded = **1,127,219** Gold rows.
  - Total: 6,879,485 eligible − 1 excluded = **6,879,484** Gold rows.

---

## 3. Leakage Prevention & Prediction Horizon

### 3.1 The Two-Hour Prediction Horizon

- **Business Objective:** "Can we identify flights likely to depart at least 15 minutes late, 2 hours before scheduled departure?"
- **Operational Prediction Cutoff:**
  $$T_{\text{cutoff}} = \text{CRS\_DEP\_TIMESTAMP} - 2\text{ hours}$$
- Every feature used by the model must be provably known and immutable at or before $T_{\text{cutoff}}$.

---

### 3.2 Target Leakage Prohibitions

The following BTS fields and derived quantities are strictly prohibited from entering feature sets:

| Prohibited Field | Leakage Mechanism | Where Discarded |
|---|---|---|
| `DEP_TIME` / `ACTUAL_DEP_TIME` | Actual departure timestamp directly determines the target. | Excluded from Silver schema. |
| `DEP_DELAY` / `DEP_DELAY_NEW` | Continuous delay duration directly equals or determines `dep_del15`. | Excluded from Silver schema. |
| `ARR_TIME` / `ARR_DELAY` | Post-flight operational outcome occurring hours after departure. | Excluded from Silver schema. |
| `TAXI_OUT` / `WHEELS_OFF` | Ground operational metrics occurring after scheduled departure. | Excluded from Silver schema. |
| `AIR_TIME` / `ACTUAL_ELAPSED_TIME` | In-flight metrics occurring post-departure. | Excluded from Silver schema. |
| `WEATHER_DELAY`, `CARRIER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` | Post-facto delay cause accounting breakdown provided by carriers. | Excluded from Silver schema. |
| `CANCELLED` / `CANCELLATION_CODE` | Cancellations are often decided after $T_{\text{cutoff}}$. | Excluded from feature vector; used only for population filtering. |
| `DIVERTED` | Diversions are determined in-flight. | Excluded from feature vector; used only for population filtering. |

---

### 3.3 Temporal Leakage Prohibitions

1. **Global Aggregations:** Never compute historical carrier or airport delay rates across the full dataset. Doing so leaks future validation and test performance into training features.
2. **Expanding / Rolling Window Policy:** Historical delay features (e.g., `origin_prior_delay_rate_7d`) must be calculated strictly over flights occurring prior to the current day:
   $$\text{Feature}(D_t) = f(\{D_k \mid k < t\})$$
3. **Chronological Split Isolation:**
   - **Train:** Flights from `2025-01-01` to `2025-08-31`.
   - **Validation:** Flights from `2025-09-01` to `2025-10-31`.
   - **Test:** Flights from `2025-11-01` to `2025-12-31`.
   - The test partition must never be included in any scaling, imputation fit, frequency encoding, or target encoding.

---

### 3.4 Weather Data Leakage & Availability Risks

Surface weather observations (METAR / ASOS) represent a significant point-in-time risk:
- **The Trap:** Joining a flight departing at 14:00 with the METAR observation at 13:52. In an operational 2-hour-ahead setting, the 13:52 observation does not exist at 12:00 ($T_{\text{cutoff}}$).
- **Operational Requirement:** If using ASOS observations, the feature pipeline must query the latest observation available **prior to $T_{\text{cutoff}}$** (e.g., the 11:52 observation for a 14:00 flight).
- **Timezone Alignment:** BTS flight times are recorded in the local airport's clock time, whereas ASOS weather observations are timestamped in UTC. Merging requires resolving local scheduled departure to UTC via `utc_local_time_variation` from `airport_reference` before joining.
- **Forecast Alternative:** The optimal long-term solution is ingesting numerical weather predictions (e.g., NOAA HRRR or RAP forecasts initialized prior to $T_{\text{cutoff}}$).

---

## 4. Required Automated Checks Before Promoting into Gold

Before promoting any data into `public.gold_features`, the pipeline must execute and pass the following automated test suite:

1. **Strict Target Integrity Test:** Assert that `dep_del15 IN (0, 1)` and no record has `is_training_eligible = FALSE`.
2. **Temporal Window Assertion:** Assert that every historical aggregation for flight date $t$ excludes data from dates $\ge t$.
3. **No Target Leakage Columns:** Scan feature schema against an explicit blacklist of post-departure terms (`delay`, `actual`, `taxi`, `air_time`, `wheels`).
4. **Timezone Offset Test:** Assert that all local-to-UTC conversions use the appropriate daylight saving time offset for 2025.
5. **Split Boundary Verification:** Assert that maximum training date $< 2025-09-01$, maximum validation date $< 2025-11-01$, and test split has zero overlap with training.

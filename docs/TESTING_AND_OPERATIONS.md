# FlightOps ML — Testing & Operational Runbook

> **Audit Status:** Verified against active Windows environment and PostgreSQL 17 container on October 20, 2026.
> **Environment:** Windows 11, PowerShell, Python 3.13 (`.venv`), Docker Desktop, PostgreSQL 17.

---

## 1. Quick Start & Operational Environment

All commands are shown for **Windows PowerShell** executed from the repository root (`flightops-ml`).

### Prerequisites
- Python 3.13 installed.
- Docker Desktop running.
- Virtual environment `.venv` created and activated.

```powershell
# 1. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 2. Verify environment sanity
python -c "import psycopg, requests, dotenv; print('Environment OK')"
```

---

## 2. Docker & PostgreSQL Operations

The PostgreSQL service runs as a container managed by Docker Compose (`compose.yaml`).

### Container Management

```powershell
# Start PostgreSQL in the background
docker compose up -d postgres

# Check container status and health
docker compose ps
# Expected: Up (healthy), port 127.0.0.1:5433->5432/tcp

# View PostgreSQL container logs
docker compose logs -f postgres

# Stop the database container (data volume is preserved)
docker compose down

# Stop and wipe database volume (CAUTION: destroys all data)
docker compose down -v
```

### Direct Database Access via `psql`

To execute queries against the container without installing PostgreSQL client tools on Windows host:

```powershell
# Interactive psql shell
docker exec -it flightops-ml-postgres-1 psql -U flightops_app -d flightops

# Execute a single SQL command
docker exec -i flightops-ml-postgres-1 psql -U flightops_app -d flightops -c "\dt+ public.*"

# Run a SQL script file from the repository
Get-Content sql/005_create_silver_flights.sql | docker exec -i flightops-ml-postgres-1 psql -U flightops_app -d flightops -v ON_ERROR_STOP=1
```

---

## 3. Automated Test Suite

### Running Pytest

The test suite runs with `pytest` using configurations in `pyproject.toml`.

```powershell
# Run the entire test suite with verbose output
.venv\Scripts\pytest -v

# Run only ingestion unit tests
.venv\Scripts\pytest tests/ingestion/ -v

# Run only Silver transformation tests
.venv\Scripts\pytest tests/transformations/ -v

# Run with short summary reporting
.venv\Scripts\pytest -ra
```

#### Verified Test Suite Output (152 Tests)
```
============================= test session starts =============================
platform win32 -- Python 3.13.x, pytest-9.1.1, pluggy-1.6.0
rootdir: flightops-ml
configfile: pyproject.toml
testpaths: tests
collected 152 items

tests/ingestion/ ...                                                     [ 42%]
tests/test_prior_leakage.py ...                                          [ 44%]
tests/test_service.py .............                                      [ 53%]
tests/test_tracking.py ...                                               [ 55%]
tests/transformations/ ...                                               [100%]

============================= 152 passed in ~18s ==============================
```

---

## 4. Code Quality, Linting & Formatting

FlightOps ML enforces code quality rules using `ruff` configured in `pyproject.toml` (rules `E`, `F`, `I`, `UP`, `B` with line length 88).

```powershell
# 1. Run linter check
.venv\Scripts\ruff check .

# 2. Automatically fix safe linter issues
.venv\Scripts\ruff check --fix .

# 3. Check code formatting without modifying files
.venv\Scripts\ruff format --check .

# 4. Apply code formatting
.venv\Scripts\ruff format .
```

*Verified Result: Zero linter errors across workspace; all files formatted.*

---

## 5. Data Pipeline Execution Procedures

### 5.1 Ingesting BTS Monthly Archives (Bronze)

The script `scripts/ingest_bts_year.py` downloads, verifies, and stages monthly BTS archives into `public.bts_flight_staging`.

```powershell
# Dry Run: Verify archives and database state for full year 2025 without downloading
python scripts/ingest_bts_year.py --year 2025 --start-month 1 --end-month 12

# Dry Run for a specific month
python scripts/ingest_bts_year.py --year 2025 --start-month 1 --end-month 1

# Execute: Download missing archives and ingest into PostgreSQL staging
python scripts/ingest_bts_year.py --year 2025 --start-month 1 --end-month 12 --execute
```

### 5.2 Transforming Bronze into Silver

The script `scripts/transform_silver_flights.py` transforms raw Bronze partitions into strongly-typed Silver flight records.

```powershell
# Dry Run: Scan Bronze and report planned transformations
python scripts/transform_silver_flights.py --year 2025

# Execute: Run transformation across all 12 months (idempotent)
python scripts/transform_silver_flights.py --year 2025 --execute

# Transform a specific range of months
python scripts/transform_silver_flights.py --year 2025 --start-month 9 --end-month 10 --execute
```

### 5.3 Automated Ingestion Verification Suite

To test end-to-end ingestion rollback, archive validation, and idempotency using pytest:

```powershell
pytest tests/ingestion/ -v
```
*Executes unit and integration tests covering archive checksums, staging COPY idempotency, and transaction rollback.*

---

## 6. Verification & Health Check Queries

Run these queries to verify the state and integrity of the database:

### Check Table Row Counts
```powershell
@'
SELECT 'bts_flight_staging' AS table_name, count(*) AS count FROM public.bts_flight_staging
UNION ALL
SELECT 'silver_flights', count(*) FROM public.silver_flights
UNION ALL
SELECT 'airport_reference', count(*) FROM public.airport_reference
UNION ALL
SELECT 'weather_observations', count(*) FROM public.weather_observations
UNION ALL
SELECT 'ingestion_runs', count(*) FROM public.ingestion_runs;
'@ | docker exec -i flightops-ml-postgres-1 psql -U flightops_app -d flightops
```
*Expected: `bts_flight_staging: 7001619`, `silver_flights: 7001619`, `airport_reference: 20295`, `weather_observations: 8760`, `ingestion_runs: 25`.*

### Verify Silver Data Integrity Invariants
```powershell
@'
SELECT
    COUNT(*) AS total_silver_rows,
    COUNT(DISTINCT (source_sha256, source_row_number)) AS unique_source_rows,
    COUNT(*) FILTER (WHERE cancelled = TRUE AND dep_del15 IS NOT NULL) AS cancelled_with_delay_label,
    COUNT(*) FILTER (WHERE is_training_eligible != (dep_del15 IS NOT NULL AND cancelled = FALSE AND diverted = FALSE)) AS eligibility_violations,
    COUNT(*) FILTER (WHERE crs_elapsed_time < 0) AS negative_elapsed_time_rows
FROM public.silver_flights;
'@ | docker exec -i flightops-ml-postgres-1 psql -U flightops_app -d flightops
```
*Expected: `total_silver_rows: 7001619`, `unique_source_rows: 7001619`, `cancelled_with_delay_label: 0`, `eligibility_violations: 0`, `negative_elapsed_time_rows: 5`.*

### Verify Chronological Split Totals
```powershell
@'
SELECT
    CASE
        WHEN flight_date < DATE '2025-09-01' THEN 'train'
        WHEN flight_date < DATE '2025-11-01' THEN 'validation'
        ELSE 'test'
    END AS dataset_split,
    MIN(flight_date) AS first_date,
    MAX(flight_date) AS last_date,
    COUNT(*) AS flights,
    ROUND(COUNT(*) FILTER (WHERE dep_del15 = 1)::NUMERIC / COUNT(*) * 100, 2) AS delay_rate
FROM public.silver_flights
WHERE is_training_eligible = TRUE
GROUP BY 1
ORDER BY MIN(flight_date);
'@ | docker exec -i flightops-ml-postgres-1 psql -U flightops_app -d flightops
```
*Expected:*
- `train`: `2025-01-01` to `2025-08-31`, 4,592,366 flights, 22.19% delay rate.
- `validation`: `2025-09-01` to `2025-10-31`, 1,159,899 flights, 18.08% delay rate.
- `test`: `2025-11-01` to `2025-12-31`, 1,127,220 flights, 23.40% delay rate.

---

## 7. Troubleshooting & Common Operational Issues

### 1. Port 5433 Conflict
- **Symptom:** Docker fails to bind `127.0.0.1:5433` (`bind: address already in use`).
- **Fix:** Another container or local service is listening on port 5433. Check using `netstat -ano | findstr 5433` and terminate the conflicting process or change `PGPORT` in `.env` and `compose.yaml`.

### 2. Missing Environment Variables
- **Symptom:** `KeyError: 'POSTGRES_DB'` or connection refused.
- **Fix:** Ensure `.env` exists in the repository root. Copy from `.env.example`:
  ```powershell
  Copy-Item .env.example .env
  ```
  Edit `.env` with appropriate local values.

### 3. Docker Healthcheck Timeout
- **Symptom:** Container stuck in `health: starting`.
- **Fix:** Inspect container logs: `docker logs flightops-ml-postgres-1`. Verify sufficient disk space and memory allocated to Docker Desktop in WSL2 settings.

---

## 8. MLflow Experiment Tracking & Model Registry Operations

MLflow is configured with a persistent SQLite database store (`sqlite:///mlflow.db`) and an artifact directory (`mlruns/`).

### Start the MLflow UI

```powershell
# Launch MLflow UI on localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to inspect runs, compare PR/ROC curves, examine SHAP feature attribution artifacts, and audit the Model Registry.

### Sync Experiments & Register Models

To re-sync or log experiments and register the champion model:

```powershell
# Sync benchmark scoreboards and holdout test runs to MLflow
python scripts/sync_experiments_to_mlflow.py
```

### Inspect via Python CLI

```powershell
python -c "
from flightops_ml.tracking import setup_tracking, get_tracking_uri
from mlflow.tracking import MlflowClient

client = MlflowClient(get_tracking_uri())
exp = client.get_experiment_by_name('flightops-delay-prediction')
print('Runs in experiment:', len(client.search_runs(exp.experiment_id)))
model = client.get_registered_model('flightops-departure-delay')
print('Model Registry:', model.name, 'Aliases:', model.aliases)
"
```

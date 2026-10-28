"""Bulk ingestion module for 2025 ASOS weather observations.

Fetches batched observations from Iowa Environmental Mesonet (IEM),
persists raw CSV artifacts with SHA-256 checksums under data/raw/weather/2025/,
and stages typed records into public.weather_observations via PostgreSQL COPY.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from flightops_ml.ingestion.loader import database_connection

LOGGER = logging.getLogger("flightops_ml.ingestion.weather_bulk_ingester")

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_USER_AGENT = (
    "FlightOps-ML-Research/1.0 (academic flight delay prediction audit)"
)

# The 20 weather observation variables stored in public.weather_observations
WEATHER_RAW_FIELDS = (
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "wxcodes",
    "metar",
)

# The 12 core observation fields requested from IEM ASOS
# (omits the 8 cloud layer tables to avoid IEM's 10x join latency bottleneck;
# cloud cover/ceiling is present verbatim in the metar text)
WEATHER_REQUEST_FIELDS = (
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "wxcodes",
    "metar",
)

TARGET_TABLE_COLUMNS = (
    "station",
    "observation_time",
    "source_sha256",
    "source_row_number",
    "ingestion_run_id",
    *(f"{field}_raw" for field in WEATHER_RAW_FIELDS),
)

DAYS_IN_MONTH_2025 = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


@dataclass(frozen=True)
class BatchIngestionResult:
    """Outcome metadata for a single weather ingestion batch."""

    batch_id: str
    year: int
    month: int | None
    stations_in_batch: list[str]
    raw_csv_path: str
    source_sha256: str
    records_seen: int
    records_inserted: int
    records_skipped: int
    duration_seconds: float
    status: str
    error_message: str | None = None


def compute_file_sha256(path: Path) -> str:
    """Calculate hex SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_iem_batch_params(
    stations: list[str],
    year: int = 2025,
    month: int | None = None,
    fields: tuple[str, ...] = WEATHER_REQUEST_FIELDS,
) -> dict[str, Any]:
    """Construct URL query parameters for an IEM multi-station query."""
    if month is not None and not 1 <= month <= 12:
        raise ValueError(f"Month must be between 1 and 12, got: {month}")

    start_month = str(month) if month is not None else "1"
    start_day = "1"
    end_month = str(month) if month is not None else "12"
    end_day = str(DAYS_IN_MONTH_2025[month]) if month is not None else "31"

    return {
        "station": sorted(set(s.strip().upper() for s in stations)),
        "data": list(fields),
        "year1": str(year),
        "month1": start_month,
        "day1": start_day,
        "hour1": "0",
        "minute1": "0",
        "year2": str(year),
        "month2": end_month,
        "day2": end_day,
        "hour2": "23",
        "minute2": "59",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": ["3", "4"],
    }


def fetch_iem_csv_stream(
    session: requests.Session,
    stations: list[str],
    year: int = 2025,
    month: int | None = None,
    fields: tuple[str, ...] = WEATHER_REQUEST_FIELDS,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: float = 120.0,
) -> str:
    """Retrieve raw CSV string from IEM ASOS service with bounded backoff."""
    params = build_iem_batch_params(stations, year=year, month=month, fields=fields)
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(
                IEM_ASOS_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch batch for {len(stations)} stations "
                        f"after {max_retries} attempts: HTTP {resp.status_code}"
                    )
                wait_time = backoff_factor**attempt
                LOGGER.warning(
                    "HTTP %d from IEM on attempt %d. Pausing %.1fs...",
                    resp.status_code,
                    attempt,
                    wait_time,
                )
                time.sleep(wait_time)
                continue
            resp.raise_for_status()

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait_time = backoff_factor**attempt
                LOGGER.warning(
                    "Request error (%s) on attempt %d. Pausing %.1fs...",
                    exc,
                    attempt,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to fetch batch after {max_retries} attempts: {exc}"
                ) from exc

    if last_exc:
        raise last_exc
    raise RuntimeError(
        f"Failed to fetch batch after {max_retries} attempts without response"
    )


def clean_weather_value(value: str | None) -> str | None:
    """Map missing marker 'M', blanks, and whitespace to None."""
    if value is None:
        return None
    val = value.strip()
    if not val or val.upper() == "M":
        return None
    return val


def parse_and_deduplicate_batch_rows(
    csv_text: str,
    source_sha256: str,
    target_year: int,
    target_month: int | None = None,
) -> list[tuple[Any, ...]]:
    """Parse raw CSV text into database-ready tuples with deterministic deduplication.

    When duplicate timestamps occur at the same minute, the record with more
    non-null weather fields is selected (preferring manual augmented METAR over
    automated sensor broadcast).
    """
    lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    if not lines:
        return []

    reader = csv.DictReader(lines)
    if not reader.fieldnames or "station" not in reader.fieldnames:
        return []

    # Map (station, observation_time) -> best candidate dictionary
    deduped_candidates: dict[tuple[str, datetime], tuple[int, dict[str, Any], int]] = {}

    for row_idx, row in enumerate(reader, start=1):
        station = row.get("station", "").strip().upper()
        valid_str = row.get("valid", "").strip()
        if not station or not valid_str:
            continue

        try:
            obs_time = datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=UTC
            )
        except ValueError:
            continue

        if obs_time.year != target_year:
            continue
        if target_month is not None and obs_time.month != target_month:
            continue

        # Count non-null fields to prioritize richer reports
        non_null_score = sum(
            1 for f in WEATHER_RAW_FIELDS if clean_weather_value(row.get(f)) is not None
        )

        # Naive UTC timestamp for Postgres timestamp without time zone
        obs_time_naive = obs_time.replace(tzinfo=None)
        key = (station, obs_time_naive)

        if key not in deduped_candidates:
            deduped_candidates[key] = (non_null_score, row, row_idx)
        else:
            existing_score, _, _ = deduped_candidates[key]
            if non_null_score > existing_score:
                deduped_candidates[key] = (non_null_score, row, row_idx)

    # Sort deterministically by (station, observation_time)
    sorted_keys = sorted(deduped_candidates.keys())
    output_rows = []

    for key in sorted_keys:
        station, obs_time_naive = key
        _, row, row_idx = deduped_candidates[key]

        cleaned_fields = tuple(
            clean_weather_value(row.get(field)) for field in WEATHER_RAW_FIELDS
        )

        output_rows.append(
            (
                station,
                obs_time_naive,
                source_sha256,
                row_idx,
                *cleaned_fields,
            )
        )

    return output_rows


def ingest_weather_batch(
    csv_text: str,
    raw_csv_path: Path,
    stations: list[str],
    year: int,
    batch_id: str,
    month: int | None = None,
    dry_run: bool = False,
) -> BatchIngestionResult:
    """Save raw CSV to disk, parse, and load batch into public.weather_observations."""
    t0 = time.time()

    # 1. Save raw CSV artifact locally with SHA-256 checksum
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(csv_text)

    checksum = compute_file_sha256(raw_csv_path)

    # 2. Parse and deduplicate rows in memory
    parsed_rows = parse_and_deduplicate_batch_rows(
        csv_text=csv_text,
        source_sha256=checksum,
        target_year=year,
        target_month=month,
    )
    records_seen = len(parsed_rows)

    if dry_run or records_seen == 0:
        elapsed = time.time() - t0
        return BatchIngestionResult(
            batch_id=batch_id,
            year=year,
            month=month,
            stations_in_batch=stations,
            raw_csv_path=str(raw_csv_path),
            source_sha256=checksum,
            records_seen=records_seen,
            records_inserted=0,
            records_skipped=records_seen,
            duration_seconds=round(elapsed, 2),
            status="dry_run" if dry_run else "empty",
        )

    # 3. Insert into public.ingestion_runs and public.weather_observations
    with database_connection() as conn:
        with conn.cursor() as cur:
            db_year = year if month is not None else None
            cur.execute(
                """
                INSERT INTO public.ingestion_runs (
                    source_name, source_filename, source_sha256,
                    source_year, source_month, status
                )
                VALUES ('IEM_ASOS', %s, %s, %s, %s, 'running')
                RETURNING run_id
                """,
                (raw_csv_path.name, checksum, db_year, month),
            )
            run_id = cur.fetchone()[0]
        conn.commit()

        try:
            with conn.cursor() as cur:
                target_sql = ", ".join(TARGET_TABLE_COLUMNS)

                cur.execute(
                    f"""
                    CREATE TEMP TABLE weather_load_tmp
                    ON COMMIT DROP AS
                    SELECT {target_sql}
                    FROM public.weather_observations
                    WITH NO DATA
                    """
                )

                # Write rows to temp table via high-performance COPY
                with cur.copy(
                    f"COPY weather_load_tmp ({target_sql}) FROM STDIN"
                ) as copy:
                    for row in parsed_rows:
                        # row: (station, obs_time, sha256, row_idx, *20_fields)
                        # target: station, obs_time, sha256, row_idx, run_id, *20_fields
                        full_row = (*row[:4], run_id, *row[4:])
                        copy.write_row(full_row)

                # Upsert into public.weather_observations
                cur.execute(
                    f"""
                    INSERT INTO public.weather_observations ({target_sql})
                    SELECT {target_sql}
                    FROM weather_load_tmp
                    ON CONFLICT (station, observation_time) DO NOTHING
                    """
                )
                inserted = cur.rowcount
                skipped = records_seen - inserted

                cur.execute(
                    """
                    UPDATE public.ingestion_runs
                    SET status = 'succeeded',
                        records_seen = %s,
                        records_inserted = %s,
                        records_skipped = %s,
                        finished_at = NOW(),
                        error_message = NULL
                    WHERE run_id = %s
                    """,
                    (records_seen, inserted, skipped, run_id),
                )

            conn.commit()
            elapsed = time.time() - t0

            return BatchIngestionResult(
                batch_id=batch_id,
                year=year,
                month=month,
                stations_in_batch=stations,
                raw_csv_path=str(raw_csv_path),
                source_sha256=checksum,
                records_seen=records_seen,
                records_inserted=inserted,
                records_skipped=skipped,
                duration_seconds=round(elapsed, 2),
                status="succeeded",
            )

        except Exception as exc:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.ingestion_runs
                    SET status = 'failed',
                        finished_at = NOW(),
                        error_message = %s
                    WHERE run_id = %s
                    """,
                    (str(exc)[:2000], run_id),
                )
            conn.commit()
            raise


def execute_weather_ingestion(
    station_ids: list[str],
    year: int = 2025,
    months: list[int] | None = None,
    annual: bool = False,
    chunk_size: int = 20,
    pause_seconds: float = 1.5,
    raw_dir: Path = Path("data/raw/weather/2025"),
    manifest_path: Path = Path("data/raw/weather/2025/.ingestion_manifest.json"),
    dry_run: bool = False,
    session: requests.Session | None = None,
) -> list[BatchIngestionResult]:
    """Execute complete batched weather ingestion across target stations."""
    managed_session = session if session is not None else requests.Session()

    # Load manifest to skip previously ingested batches
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    unique_stations = sorted(set(s.strip().upper() for s in station_ids))
    station_chunks = [
        unique_stations[i : i + chunk_size]
        for i in range(0, len(unique_stations), chunk_size)
    ]

    batch_results: list[BatchIngestionResult] = []

    if annual:
        total_batches = len(station_chunks)
        LOGGER.info(
            "Starting annual weather ingestion: %d stations, %d chunks (%d batches)",
            len(unique_stations),
            len(station_chunks),
            total_batches,
        )

        for chunk_idx, chunk in enumerate(station_chunks, start=1):
            batch_id = f"{year}_annual_chunk{chunk_idx:02d}"

            if batch_id in manifest and manifest[batch_id].get("status") == "succeeded":
                LOGGER.info(
                    "[%d/%d] Batch %s already completed. Skipping.",
                    chunk_idx,
                    total_batches,
                    batch_id,
                )
                continue

            LOGGER.info(
                "[%d/%d] Processing annual batch %s (%d stations: %s...)",
                chunk_idx,
                total_batches,
                batch_id,
                len(chunk),
                chunk[0],
            )

            raw_file_name = f"weather_{year}_annual_chunk{chunk_idx:02d}.csv"
            raw_path = raw_dir / raw_file_name

            time.sleep(pause_seconds)

            try:
                csv_text = fetch_iem_csv_stream(
                    session=managed_session,
                    stations=chunk,
                    year=year,
                    month=None,
                )

                result = ingest_weather_batch(
                    csv_text=csv_text,
                    raw_csv_path=raw_path,
                    stations=chunk,
                    year=year,
                    month=None,
                    batch_id=batch_id,
                    dry_run=dry_run,
                )
                batch_results.append(result)

                manifest[batch_id] = {
                    "batch_id": result.batch_id,
                    "year": result.year,
                    "month": result.month,
                    "station_count": len(result.stations_in_batch),
                    "raw_csv_path": result.raw_csv_path,
                    "source_sha256": result.source_sha256,
                    "records_seen": result.records_seen,
                    "records_inserted": result.records_inserted,
                    "records_skipped": result.records_skipped,
                    "duration_seconds": result.duration_seconds,
                    "status": result.status,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                raw_dir.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                LOGGER.info(
                    "[%d/%d] Batch %s finished in %.1fs: "
                    "%d seen, %d inserted, %d skipped",
                    chunk_idx,
                    total_batches,
                    batch_id,
                    result.duration_seconds,
                    result.records_seen,
                    result.records_inserted,
                    result.records_skipped,
                )

            except Exception as exc:
                LOGGER.error("Batch %s failed: %s", batch_id, exc)
                manifest[batch_id] = {
                    "batch_id": batch_id,
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                raw_dir.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                raise

        return batch_results

    # Monthly mode
    if months is None:
        target_months = list(range(1, 13))
    else:
        target_months = sorted(set(months))

    total_batches = len(target_months) * len(station_chunks)
    LOGGER.info(
        "Starting monthly weather ingestion: %d stations, %d months, "
        "%d chunks/month (%d total batches)",
        len(unique_stations),
        len(target_months),
        len(station_chunks),
        total_batches,
    )

    batch_counter = 0

    for m in target_months:
        for chunk_idx, chunk in enumerate(station_chunks, start=1):
            batch_counter += 1
            batch_id = f"2025_m{m:02d}_chunk{chunk_idx:02d}"

            # Check if batch is already completed in manifest
            if batch_id in manifest and manifest[batch_id].get("status") == "succeeded":
                LOGGER.info(
                    "[%d/%d] Batch %s already completed. Skipping.",
                    batch_counter,
                    total_batches,
                    batch_id,
                )
                continue

            LOGGER.info(
                "[%d/%d] Processing batch %s (%d stations: %s...)",
                batch_counter,
                total_batches,
                batch_id,
                len(chunk),
                chunk[0],
            )

            raw_file_name = f"weather_2025_m{m:02d}_chunk{chunk_idx:02d}.csv"
            raw_path = raw_dir / raw_file_name

            # Polite delay between requests
            time.sleep(pause_seconds)

            try:
                csv_text = fetch_iem_csv_stream(
                    session=managed_session,
                    stations=chunk,
                    year=year,
                    month=m,
                )

                result = ingest_weather_batch(
                    csv_text=csv_text,
                    raw_csv_path=raw_path,
                    stations=chunk,
                    year=year,
                    month=m,
                    batch_id=batch_id,
                    dry_run=dry_run,
                )
                batch_results.append(result)

                # Update manifest
                manifest[batch_id] = {
                    "batch_id": result.batch_id,
                    "year": result.year,
                    "month": result.month,
                    "station_count": len(result.stations_in_batch),
                    "raw_csv_path": result.raw_csv_path,
                    "source_sha256": result.source_sha256,
                    "records_seen": result.records_seen,
                    "records_inserted": result.records_inserted,
                    "records_skipped": result.records_skipped,
                    "duration_seconds": result.duration_seconds,
                    "status": result.status,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                raw_dir.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                LOGGER.info(
                    "[%d/%d] Batch %s finished in %.1fs: "
                    "%d seen, %d inserted, %d skipped",
                    batch_counter,
                    total_batches,
                    batch_id,
                    result.duration_seconds,
                    result.records_seen,
                    result.records_inserted,
                    result.records_skipped,
                )

            except Exception as exc:
                LOGGER.error("Batch %s failed: %s", batch_id, exc)
                manifest[batch_id] = {
                    "batch_id": batch_id,
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                raw_dir.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                raise

    return batch_results

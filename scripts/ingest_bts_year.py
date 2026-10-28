import argparse
import csv
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import requests

from flightops_ml.ingestion.archive import validate_bts_archive
from flightops_ml.ingestion.downloader import download_month
from flightops_ml.ingestion.loader import (
    SOURCE_COLUMNS,
    database_connection,
    load_month,
    sha256_file,
)

LOGGER = logging.getLogger("flightops_ml.ingest_bts_year")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "bts"


def parse_flight_date(value: str) -> datetime:
    """Parse the date formats observed in BTS monthly CSV files."""
    value = value.strip()
    if "/" in value:
        try:
            return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")
        except ValueError as exc:
            raise ValueError(f"Unrecognized BTS FL_DATE value: {value!r}") from exc

    elif "-" in value:
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Unrecognized BTS FL_DATE value: {value!r}") from exc

    raise ValueError(f"Unrecognized BTS FL_DATE value: {value!r}")


def validate_month_contents(
    archive_path: Path,
    year: int,
    month: int,
) -> int:
    """Validate the CSV schema, period fields, and every flight date."""
    csv_member = validate_bts_archive(archive_path)
    rows_seen = 0
    first_date = None
    last_date = None
    distinct_dates = set()

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(csv_member) as binary_file:
            with io.TextIOWrapper(
                binary_file,
                encoding="utf-8-sig",
                newline="",
            ) as text_file:
                reader = csv.DictReader(text_file)

                if (
                    reader.fieldnames is None
                    or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or set(reader.fieldnames) != set(SOURCE_COLUMNS)
                ):
                    raise ValueError(
                        f"{archive_path.name}: CSV columns do not match "
                        "the expected BTS schema"
                    )

                for row_number, row in enumerate(reader, start=1):
                    if None in row or any(
                        row.get(column) is None for column in SOURCE_COLUMNS
                    ):
                        raise ValueError(
                            f"{archive_path.name}: malformed CSV row {row_number}"
                        )

                    try:
                        row_year = int(float(row["YEAR"]))
                        row_month = int(float(row["MONTH"]))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{archive_path.name}: invalid YEAR/MONTH "
                            f"at row {row_number}"
                        ) from exc

                    if (row_year, row_month) != (year, month):
                        raise ValueError(
                            f"{archive_path.name}: unexpected YEAR/MONTH "
                            f"at row {row_number}: {row_year}/{row_month}"
                        )

                    try:
                        flight_date = parse_flight_date(row["FL_DATE"]).date()
                    except ValueError as exc:
                        raise ValueError(
                            f"{archive_path.name}: invalid FL_DATE at row {row_number}"
                        ) from exc

                    if flight_date.year != year or flight_date.month != month:
                        raise ValueError(
                            f"{archive_path.name}: flight date {flight_date} "
                            f"is outside {year}-{month:02d}"
                        )

                    rows_seen += 1
                    distinct_dates.add(flight_date)
                    first_date = (
                        flight_date
                        if first_date is None
                        else min(first_date, flight_date)
                    )
                    last_date = (
                        flight_date
                        if last_date is None
                        else max(last_date, flight_date)
                    )

    if rows_seen == 0:
        raise ValueError(f"{archive_path.name}: archive contains no flight rows")

    expected_days = (
        datetime(year + (month == 12), month % 12 + 1, 1).date()
        - datetime(year, month, 1).date()
    ).days

    if len(distinct_dates) != expected_days:
        raise ValueError(
            f"{archive_path.name}: expected {expected_days} distinct calendar "
            f"dates, found {len(distinct_dates)}"
        )

    if first_date.day != 1 or last_date.day != expected_days:
        raise ValueError(
            f"{archive_path.name}: date coverage is {first_date} through "
            f"{last_date}, not the full calendar month"
        )

    LOGGER.info(
        "%s validated: %s rows, %s through %s",
        archive_path.name,
        f"{rows_seen:,}",
        first_date,
        last_date,
    )
    return rows_seen


def get_database_state(
    checksum: str,
    year: int,
    month: int,
) -> tuple[int, int, bool]:
    """Return staged row count, audited row count, and completeness status."""
    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    MIN(source_row_number),
                    MAX(source_row_number)
                FROM public.bts_flight_staging
                WHERE source_sha256 = %s
                """,
                (checksum,),
            )
            staged_rows, min_row_number, max_row_number = cur.fetchone()

            cur.execute(
                """
                SELECT
                    records_seen,
                    records_inserted,
                    records_skipped
                FROM public.ingestion_runs
                WHERE source_name = 'BTS'
                  AND source_sha256 = %s
                  AND source_year = %s
                  AND source_month = %s
                  AND status = 'succeeded'
                ORDER BY run_id DESC
                LIMIT 1
                """,
                (checksum, year, month),
            )
            successful_run = cur.fetchone()

    if successful_run is None:
        return staged_rows, 0, False

    audited_rows, records_inserted, records_skipped = successful_run

    audit_totals_match = (
        audited_rows > 0 and records_inserted + records_skipped == audited_rows
    )
    row_range_matches = (
        staged_rows == audited_rows
        and min_row_number == 1
        and max_row_number == audited_rows
    )

    return staged_rows, audited_rows, audit_totals_match and row_range_matches


def process_month(
    session: requests.Session,
    raw_root: Path,
    year: int,
    month: int,
    execute: bool,
) -> str:
    """Validate, download if needed, and ingest one monthly archive."""
    archive_path = raw_root / str(year) / f"{year}-{month:02d}.zip"

    if not archive_path.exists():
        if not execute:
            LOGGER.info("%s: WOULD DOWNLOAD, VALIDATE, AND LOAD", archive_path.name)
            return "planned"

        LOGGER.info("%s: downloading", archive_path.name)
        download_month(session, year, month, raw_root)

    validate_month_contents(archive_path, year, month)
    checksum = sha256_file(archive_path)

    staged_rows, audited_rows, already_loaded = get_database_state(
        checksum,
        year,
        month,
    )

    if already_loaded:
        LOGGER.info(
            "%s: SKIP — verified successful audit and %s staged rows",
            archive_path.name,
            f"{staged_rows:,}",
        )
        return "skipped"

    if not execute:
        LOGGER.info(
            "%s: WOULD LOAD — staged rows=%s, last successful audit rows=%s",
            archive_path.name,
            f"{staged_rows:,}",
            f"{audited_rows:,}",
        )
        return "planned"

    LOGGER.info("%s: loading into PostgreSQL", archive_path.name)
    result = load_month(archive_path, year, month)

    expected_rows = result["records_seen"]
    staged_after, audited_after, complete_after = get_database_state(
        checksum,
        year,
        month,
    )

    if staged_after != expected_rows or not complete_after:
        raise RuntimeError(
            f"{archive_path.name}: post-load integrity check failed; "
            f"expected_rows={expected_rows}, "
            f"staged_rows={staged_after}, "
            f"audited_rows={audited_after}, "
            f"complete={complete_after}"
        )

    LOGGER.info(
        "%s: SUCCESS — run_id=%s, seen=%s, inserted=%s, skipped=%s",
        archive_path.name,
        result["run_id"],
        f"{result['records_seen']:,}",
        f"{result['records_inserted']:,}",
        f"{result['records_skipped']:,}",
    )
    return "loaded"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, validate, and ingest monthly BTS flight data."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download missing archives and ingest data. Default is dry run.",
    )
    args = parser.parse_args()

    if not 1987 <= args.year <= 2026:
        parser.error("--year must be between 1987 and 2026")
    if not 1 <= args.start_month <= args.end_month <= 12:
        parser.error("Require 1 <= --start-month <= --end-month <= 12")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mode = "EXECUTE" if args.execute else "DRY RUN"
    LOGGER.info(
        "%s: processing %s months %02d through %02d",
        mode,
        args.year,
        args.start_month,
        args.end_month,
    )

    results = {"loaded": 0, "skipped": 0, "planned": 0, "failed": 0}

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "FlightOpsML/1.0 (research data ingestion)"}
        )

        for month in range(args.start_month, args.end_month + 1):
            try:
                status = process_month(
                    session,
                    args.raw_root,
                    args.year,
                    month,
                    args.execute,
                )
                results[status] += 1
            except Exception:
                results["failed"] += 1
                LOGGER.exception("%s-%02d FAILED", args.year, month)

    LOGGER.info("FINAL SUMMARY: %s", results)
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

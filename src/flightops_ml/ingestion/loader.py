import csv
import hashlib
import io
import os
import zipfile
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from flightops_ml.ingestion.archive import validate_bts_archive

SOURCE_COLUMNS = (
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN_AIRPORT_ID",
    "ORIGIN",
    "DEST_AIRPORT_ID",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DEL15",
    "CRS_ARR_TIME",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
)

STAGING_COLUMNS = (
    "year_raw",
    "quarter_raw",
    "month_raw",
    "day_of_month_raw",
    "day_of_week_raw",
    "flight_date_raw",
    "op_unique_carrier_raw",
    "origin_airport_id_raw",
    "origin_raw",
    "dest_airport_id_raw",
    "dest_raw",
    "crs_dep_time_raw",
    "dep_del15_raw",
    "crs_arr_time_raw",
    "cancelled_raw",
    "diverted_raw",
    "crs_elapsed_time_raw",
    "distance_raw",
)

TARGET_COLUMNS = (
    "source_sha256",
    "source_row_number",
    "ingestion_run_id",
    *STAGING_COLUMNS,
)


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 checksum without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def database_connection():
    """Connect using PostgreSQL settings from the environment."""
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5433")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def load_month(archive_path: Path, year: int, month: int) -> dict:
    """Validate and bulk-load one BTS archive into PostgreSQL staging."""
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    csv_member = validate_bts_archive(archive_path)
    checksum = sha256_file(archive_path)

    target_column_sql = ", ".join(TARGET_COLUMNS)

    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.ingestion_runs (
                    source_name,
                    source_filename,
                    source_sha256,
                    source_year,
                    source_month,
                    status
                )
                VALUES ('BTS', %s, %s, %s, %s, 'running')
                RETURNING run_id
                """,
                (archive_path.name, checksum, year, month),
            )
            run_id = cur.fetchone()[0]

        # Preserve the running record if the ingestion transaction fails.
        conn.commit()

        records_seen = 0
        records_inserted = 0

        try:
            with conn.cursor() as cur:
                # Copy the target column types, but not its indexes or constraints.
                cur.execute(
                    f"""
                    CREATE TEMP TABLE flightops_bts_load_tmp
                    ON COMMIT DROP AS
                    SELECT {target_column_sql}
                    FROM public.bts_flight_staging
                    WITH NO DATA
                    """
                )

                copy_sql = (
                    f"COPY flightops_bts_load_tmp ({target_column_sql}) FROM STDIN"
                )

                with zipfile.ZipFile(archive_path) as archive:
                    with archive.open(csv_member) as binary_file:
                        with io.TextIOWrapper(
                            binary_file,
                            encoding="utf-8-sig",
                            newline="",
                        ) as text_file:
                            reader = csv.DictReader(text_file)

                            if reader.fieldnames is None:
                                raise ValueError("CSV header is missing")

                            if len(reader.fieldnames) != len(
                                set(reader.fieldnames)
                            ) or set(reader.fieldnames) != set(SOURCE_COLUMNS):
                                raise ValueError(
                                    "CSV columns do not match the expected BTS schema"
                                )

                            with cur.copy(copy_sql) as copy:
                                for row_number, row in enumerate(reader, start=1):
                                    records_seen += 1

                                    if None in row or any(
                                        row.get(col) is None for col in SOURCE_COLUMNS
                                    ):
                                        raise ValueError(
                                            f"Malformed CSV record at row {row_number}"
                                        )

                                    if (
                                        row["YEAR"].strip() != f"{year}.00"
                                        or row["MONTH"].strip() != f"{month}.00"
                                    ):
                                        # Numeric formatting may vary in source files.
                                        try:
                                            row_year = int(float(row["YEAR"]))
                                            row_month = int(float(row["MONTH"]))
                                        except (ValueError, TypeError) as exc:
                                            message = (
                                                f"Invalid period at row {row_number}"
                                            )

                                            raise ValueError(message) from exc

                                        if row_year != year or row_month != month:
                                            raise ValueError(
                                                "Archive contains a row outside the "
                                                f"requested period at row {row_number}"
                                            )

                                    copy.write_row(
                                        (
                                            checksum,
                                            row_number,
                                            run_id,
                                            *(row[col] for col in SOURCE_COLUMNS),
                                        )
                                    )

                # Merge atomically and ignore rows already loaded from this archive.
                cur.execute(
                    f"""
                    INSERT INTO public.bts_flight_staging ({target_column_sql})
                    SELECT {target_column_sql}
                    FROM flightops_bts_load_tmp
                    ON CONFLICT (source_sha256, source_row_number)
                    DO NOTHING
                    """
                )
                records_inserted = cur.rowcount

                cur.execute(
                    """
                    UPDATE public.ingestion_runs
                    SET status = 'succeeded',
                        records_seen = %s,
                        records_inserted = %s,
                        records_skipped = %s,
                        finished_at = NOW()
                    WHERE run_id = %s
                    """,
                    (
                        records_seen,
                        records_inserted,
                        records_seen - records_inserted,
                        run_id,
                    ),
                )

            conn.commit()

        except Exception as exc:
            conn.rollback()

            # The audit row was committed before the data transaction began.
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.ingestion_runs
                    SET status = 'failed',
                        records_seen = %s,
                        finished_at = NOW(),
                        error_message = %s
                    WHERE run_id = %s
                    """,
                    (records_seen, str(exc)[:2000], run_id),
                )

            conn.commit()
            raise

    return {
        "run_id": run_id,
        "source_sha256": checksum,
        "records_seen": records_seen,
        "records_inserted": records_inserted,
        "records_skipped": records_seen - records_inserted,
    }

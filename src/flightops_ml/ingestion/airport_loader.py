import csv
from datetime import date, datetime
from pathlib import Path

from flightops_ml.ingestion.loader import database_connection
from flightops_ml.ingestion.weather_loader import file_sha256

EXPECTED_COLUMNS = (
    "AIRPORT_SEQ_ID",
    "AIRPORT_ID",
    "AIRPORT",
    "DISPLAY_AIRPORT_NAME",
    "LATITUDE",
    "LONGITUDE",
    "UTC_LOCAL_TIME_VARIATION",
    "AIRPORT_START_DATE",
    "AIRPORT_THRU_DATE",
    "AIRPORT_IS_CLOSED",
    "AIRPORT_IS_LATEST",
)

TARGET_COLUMNS = (
    "airport_seq_id",
    "airport_id",
    "airport_code",
    "airport_name",
    "display_airport_name",
    "latitude",
    "longitude",
    "utc_local_time_variation",
    "airport_start_date",
    "airport_thru_date",
    "is_closed",
    "is_latest",
    "source_sha256",
    "source_row_number",
    "ingestion_run_id",
)


def optional_text(value: str | None) -> str | None:
    """Convert blank text to NULL while preserving nonblank text."""
    if value is None or not value.strip():
        return None
    return value.strip()


def required_integer(value: str | None, field: str, row_number: int) -> int:
    """Parse a required integer and report the source row on failure."""
    cleaned = optional_text(value)
    if cleaned is None:
        raise ValueError(f"Missing {field} at row {row_number}")

    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for {field} at row {row_number}: {cleaned!r}"
        ) from exc


def optional_coordinate(
    value: str | None, field: str, row_number: int, minimum: float, maximum: float
) -> float | None:
    """Parse an optional coordinate and enforce its valid range."""
    cleaned = optional_text(value)
    if cleaned is None:
        return None

    try:
        coordinate = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid {field} at row {row_number}: {cleaned!r}") from exc

    if not minimum <= coordinate <= maximum:
        raise ValueError(f"{field} outside [{minimum}, {maximum}] at row {row_number}")

    return coordinate


def optional_date(value: str | None, field: str, row_number: int) -> date | None:
    """Parse supported BTS date formats, preserving blank dates as NULL."""
    cleaned = optional_text(value)
    if cleaned is None:
        return None

    for date_format in ("%Y-%m-%d", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue

    raise ValueError(f"Invalid date for {field} at row {row_number}: {cleaned!r}")


def required_flag(value: str | None, field: str, row_number: int) -> bool:
    """Parse a required BTS flag represented as Y/N or 1/0."""
    cleaned = optional_text(value)
    if cleaned is None:
        raise ValueError(f"Missing {field} at row {row_number}")

    normalized = cleaned.upper()
    flag_values = {
        "Y": True,
        "N": False,
        "1": True,
        "0": False,
    }

    if normalized not in flag_values:
        raise ValueError(
            f"Invalid {field} at row {row_number}: "
            f"expected Y, N, 1, or 0, got {cleaned!r}"
        )

    return flag_values[normalized]


def read_airport_reference(path: Path) -> tuple[str, list[tuple]]:
    """Read and validate a BTS airport reference CSV without database writes."""
    path = Path(path)
    checksum = file_sha256(path)
    rows = []
    seen_sequence_ids = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or tuple(reader.fieldnames) != EXPECTED_COLUMNS
        ):
            raise ValueError(
                "CSV columns do not match the expected BTS airport reference schema"
            )

        for row_number, row in enumerate(reader, start=1):
            if None in row or any(
                row.get(column) is None for column in EXPECTED_COLUMNS
            ):
                raise ValueError(f"Malformed CSV record at row {row_number}")

            sequence_id = required_integer(
                row["AIRPORT_SEQ_ID"], "AIRPORT_SEQ_ID", row_number
            )
            if sequence_id in seen_sequence_ids:
                raise ValueError(
                    f"Duplicate AIRPORT_SEQ_ID at row {row_number}: {sequence_id}"
                )
            seen_sequence_ids.add(sequence_id)

            airport_id = required_integer(row["AIRPORT_ID"], "AIRPORT_ID", row_number)
            airport_code = optional_text(row["AIRPORT"])
            if airport_code is None:
                raise ValueError(f"Missing AIRPORT code at row {row_number}")

            latitude = optional_coordinate(
                row["LATITUDE"], "LATITUDE", row_number, -90, 90
            )
            longitude = optional_coordinate(
                row["LONGITUDE"], "LONGITUDE", row_number, -180, 180
            )
            if (latitude is None) != (longitude is None):
                raise ValueError(
                    "Latitude and longitude must both be present or both be blank "
                    f"at row {row_number}"
                )

            start_date = optional_date(
                row["AIRPORT_START_DATE"], "AIRPORT_START_DATE", row_number
            )
            if start_date is None:
                raise ValueError(f"Missing AIRPORT_START_DATE at row {row_number}")

            end_date = optional_date(
                row["AIRPORT_THRU_DATE"], "AIRPORT_THRU_DATE", row_number
            )
            if end_date is not None and end_date < start_date:
                raise ValueError(
                    f"AIRPORT_THRU_DATE precedes start date at row {row_number}"
                )

            rows.append(
                (
                    sequence_id,
                    airport_id,
                    airport_code,
                    None,
                    optional_text(row["DISPLAY_AIRPORT_NAME"]),
                    latitude,
                    longitude,
                    optional_text(row["UTC_LOCAL_TIME_VARIATION"]),
                    start_date,
                    end_date,
                    required_flag(
                        row["AIRPORT_IS_CLOSED"], "AIRPORT_IS_CLOSED", row_number
                    ),
                    required_flag(
                        row["AIRPORT_IS_LATEST"], "AIRPORT_IS_LATEST", row_number
                    ),
                    checksum,
                    row_number,
                )
            )

    if not rows:
        raise ValueError("Airport reference CSV contains no records")

    latest_ids = [row[1] for row in rows if row[11]]
    if len(latest_ids) != len(set(latest_ids)):
        raise ValueError("Multiple latest records found for the same AIRPORT_ID")

    return checksum, rows


def load_airport_reference(path: Path) -> dict:
    """Validate and load a BTS airport reference snapshot."""
    path = Path(path)
    checksum, rows = read_airport_reference(path)

    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.ingestion_runs (
                    source_name, source_filename, source_sha256,
                    source_year, source_month, status
                )
                VALUES ('BTS_AIRPORT_REFERENCE', %s, %s, NULL, NULL, 'running')
                RETURNING run_id
                """,
                (path.name, checksum),
            )
            run_id = cur.fetchone()[0]
        conn.commit()

        try:
            with conn.cursor() as cur:
                target_sql = ", ".join(TARGET_COLUMNS)

                cur.execute(
                    f"""
                    CREATE TEMP TABLE airport_reference_load_tmp
                    ON COMMIT DROP AS
                    SELECT {target_sql}
                    FROM public.airport_reference
                    WITH NO DATA
                    """
                )

                with cur.copy(
                    f"COPY airport_reference_load_tmp ({target_sql}) FROM STDIN"
                ) as copy:
                    for row in rows:
                        copy.write_row((*row, run_id))

                cur.execute(
                    f"""
                    INSERT INTO public.airport_reference ({target_sql})
                    SELECT {target_sql}
                    FROM airport_reference_load_tmp
                    ON CONFLICT (source_sha256, source_row_number) DO NOTHING
                    """
                )
                inserted = cur.rowcount
                seen = len(rows)
                skipped = seen - inserted

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
                    (seen, inserted, skipped, run_id),
                )

            conn.commit()

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

    return {
        "run_id": run_id,
        "source_filename": path.name,
        "source_sha256": checksum,
        "records_seen": seen,
        "records_inserted": inserted,
        "records_skipped": skipped,
    }

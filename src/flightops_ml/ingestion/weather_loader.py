import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from flightops_ml.ingestion.loader import database_connection

EXPECTED_COLUMNS = {
    "station",
    "valid",
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
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
    "peak_wind_gust",
    "peak_wind_drct",
    "peak_wind_time",
    "feel",
    "metar",
    "snowdepth",
}

WEATHER_FIELDS = (
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

TARGET_COLUMNS = (
    "station",
    "observation_time",
    "source_sha256",
    "source_row_number",
    "ingestion_run_id",
    *(f"{field}_raw" for field in WEATHER_FIELDS),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_observation_time(value: str) -> datetime:
    """Parse IEM's UTC timestamp without using the machine's local timezone."""
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M")
    return parsed.replace(tzinfo=UTC).replace(tzinfo=None)


def clean_value(value: str | None) -> str | None:
    """Preserve raw values, mapping blanks and the M missing marker to NULL."""
    if value is None or not value.strip() or value.strip().upper() == "M":
        return None
    return value.strip()


def load_weather_file(path: Path, year: int, month: int) -> dict:
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    path = Path(path)
    checksum = file_sha256(path)
    expected_station = path.stem.upper()
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or set(reader.fieldnames) != EXPECTED_COLUMNS
        ):
            raise ValueError("CSV columns do not match the expected IEM weather schema")

        for row_number, row in enumerate(reader, start=1):
            if None in row or any(row.get(col) is None for col in EXPECTED_COLUMNS):
                raise ValueError(f"Malformed CSV record at row {row_number}")

            station = row["station"].strip().upper()
            if not station or station != expected_station:
                raise ValueError(
                    f"Station mismatch at row {row_number}: "
                    f"expected {expected_station}, got {station!r}"
                )

            observed_at = parse_observation_time(row["valid"])
            if observed_at.year != year or observed_at.month != month:
                raise ValueError(
                    f"Observation outside requested period at row {row_number}"
                )

            rows.append(
                (
                    station,
                    observed_at,
                    checksum,
                    row_number,
                    *(clean_value(row[field]) for field in WEATHER_FIELDS),
                )
            )

    if not rows:
        raise ValueError("Weather CSV contains no observations")

    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.ingestion_runs (
                    source_name, source_filename, source_sha256,
                    source_year, source_month, status
                )
                VALUES ('IEM_ASOS', %s, %s, %s, %s, 'running')
                RETURNING run_id
                """,
                (path.name, checksum, year, month),
            )
            run_id = cur.fetchone()[0]
        conn.commit()

        try:
            with conn.cursor() as cur:
                target_sql = ", ".join(TARGET_COLUMNS)

                cur.execute(
                    f"""
                    CREATE TEMP TABLE weather_load_tmp
                    ON COMMIT DROP AS
                    SELECT {target_sql}
                    FROM public.weather_observations
                    WITH NO DATA
                    """
                )

                with cur.copy(
                    f"COPY weather_load_tmp ({target_sql}) FROM STDIN"
                ) as copy:
                    for row in rows:
                        copy.write_row((*row[:4], run_id, *row[4:]))

                cur.execute(
                    f"""
                    INSERT INTO public.weather_observations ({target_sql})
                    SELECT {target_sql}
                    FROM weather_load_tmp
                    ON CONFLICT (station, observation_time) DO NOTHING
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
        "source_sha256": checksum,
        "records_seen": seen,
        "records_inserted": inserted,
        "records_skipped": skipped,
    }

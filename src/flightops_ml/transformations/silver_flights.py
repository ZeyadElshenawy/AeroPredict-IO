import logging
from datetime import date, datetime, time, timedelta

LOGGER = logging.getLogger("flightops_ml.transformations.silver_flights")

TARGET_COLUMNS = (
    "source_sha256",
    "source_row_number",
    "ingestion_run_id",
    "ingested_at",
    "flight_date",
    "year",
    "quarter",
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
    "origin_airport_id",
    "origin",
    "dest_airport_id",
    "dest",
    "crs_dep_time_raw",
    "crs_arr_time_raw",
    "crs_dep_time_str",
    "crs_arr_time_str",
    "crs_dep_time",
    "crs_arr_time",
    "crs_dep_timestamp",
    "crs_elapsed_time",
    "distance",
    "cancelled",
    "diverted",
    "dep_del15",
    "is_training_eligible",
)


def parse_flight_date(value: str) -> date:
    """Parse BTS FL_DATE values supporting US-style and ISO formats."""
    value = value.strip()
    if "/" in value:
        parts = value.split()
        date_part = parts[0]
        segments = date_part.split("/")
        if len(segments) != 3:
            raise ValueError(f"Unrecognized flight date format: {value!r}")
        try:
            month, day, year = (int(s) for s in segments)
            return date(year, month, day)
        except ValueError as exc:
            raise ValueError(f"Invalid flight date values: {value!r}") from exc

    if "-" in value:
        parts = value.split()
        try:
            return date.fromisoformat(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid ISO flight date: {value!r}") from exc

    raise ValueError(f"Unrecognized flight date format: {value!r}")


def normalize_hhmm_time(value: str, field_name: str = "time") -> str:
    """Validate and left-pad an HHMM time string to 4 digits."""
    value = value.strip()
    if not value.isdigit() or not (1 <= len(value) <= 4):
        raise ValueError(f"Invalid {field_name} numeric format: {value!r}")

    padded = value.zfill(4)
    hour = int(padded[:2])
    minute = int(padded[2:])

    if hour > 24 or minute > 59:
        raise ValueError(f"Out of range {field_name} (hour/minute): {value!r}")
    if hour == 24 and minute != 0:
        raise ValueError(f"24:xx is invalid for {field_name}: {value!r}")

    return padded


def parse_clock_time(value: str, field_name: str = "time") -> time:
    """Parse normalized HHMM time into clock time (00:00 to 23:59).

    2400 is normalized to 00:00:00 to prevent Python/psycopg out-of-range errors.
    """
    normalized = normalize_hhmm_time(value, field_name)
    hour = int(normalized[:2])
    minute = int(normalized[2:])

    if hour == 24:
        return time(0, 0, 0)
    return time(hour, minute, 0)


def build_crs_dep_timestamp(flight_date: date, time_str: str) -> datetime:
    """Construct scheduled departure timestamp.

    Explicitly treats 2400 as midnight (00:00:00) on the following day
    while retaining flight_date as the scheduled operational date.
    """
    normalized = normalize_hhmm_time(time_str, "CRS_DEP_TIME")
    hour = int(normalized[:2])
    minute = int(normalized[2:])

    if hour == 24:
        next_day = flight_date + timedelta(days=1)
        return datetime.combine(next_day, time(0, 0, 0))

    return datetime.combine(flight_date, time(hour, minute, 0))


def clean_flag(value: str | None, field_name: str) -> bool:
    """Parse numeric and text boolean flags."""
    if value is None or not value.strip():
        raise ValueError(f"Missing required flag {field_name}")

    val = value.strip().upper()
    if val in ("1.00", "1.0", "1", "Y", "TRUE"):
        return True
    if val in ("0.00", "0.0", "0", "N", "FALSE"):
        return False

    raise ValueError(f"Invalid flag value for {field_name}: {value!r}")


def clean_dep_del15(value: str | None, cancelled: bool) -> int | None:
    """Parse binary delay label.

    Preserves NULL for cancelled flights; never imputes cancelled flights to zero.
    """
    if cancelled:
        return None

    if value is None or not value.strip():
        return None

    try:
        num = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid DEP_DEL15 format: {value!r}") from exc

    if num == 1.0:
        return 1
    if num == 0.0:
        return 0

    raise ValueError(f"Unexpected DEP_DEL15 label value: {value!r}")


def compute_training_eligibility(
    dep_del15: int | None,
    cancelled: bool,
    diverted: bool,
) -> bool:
    """Determine whether a flight outcome is eligible for training."""
    return dep_del15 is not None and not cancelled and not diverted


def parse_elapsed_time(value: str | None) -> float | None:
    """Parse scheduled elapsed time in minutes, preserving missing as NULL."""
    if value is None or not value.strip():
        return None

    try:
        num = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid CRS_ELAPSED_TIME: {value!r}") from exc

    return num


def parse_distance(value: str | None) -> float:
    """Parse flight route distance in statute miles."""
    if value is None or not value.strip():
        raise ValueError("Missing required DISTANCE")

    try:
        num = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid DISTANCE: {value!r}") from exc

    if num < 0:
        raise ValueError(f"Negative DISTANCE: {value!r}")
    return num


def transform_flight_row(raw: dict) -> dict:
    """Validate and transform a single Bronze staging dictionary into Silver types."""
    flight_date = parse_flight_date(raw["flight_date_raw"])
    year = int(float(raw["year_raw"]))
    quarter = int(float(raw["quarter_raw"]))
    month = int(float(raw["month_raw"]))
    day_of_month = int(float(raw["day_of_month_raw"]))
    day_of_week = int(float(raw["day_of_week_raw"]))

    if (flight_date.year, flight_date.month, flight_date.day) != (
        year,
        month,
        day_of_month,
    ):
        raise ValueError(
            f"Calendar field mismatch: flight_date={flight_date} vs "
            f"year={year}, month={month}, day={day_of_month}"
        )

    carrier = raw["op_unique_carrier_raw"].strip()
    if not carrier:
        raise ValueError("Missing OP_UNIQUE_CARRIER")

    origin = raw["origin_raw"].strip()
    dest = raw["dest_raw"].strip()
    if len(origin) != 3 or len(dest) != 3:
        raise ValueError(f"Invalid airport code length: origin={origin}, dest={dest}")

    dep_raw = raw["crs_dep_time_raw"].strip()
    arr_raw = raw["crs_arr_time_raw"].strip()
    dep_str = normalize_hhmm_time(dep_raw, "CRS_DEP_TIME")
    arr_str = normalize_hhmm_time(arr_raw, "CRS_ARR_TIME")
    dep_time = parse_clock_time(dep_raw, "CRS_DEP_TIME")
    arr_time = parse_clock_time(arr_raw, "CRS_ARR_TIME")
    dep_timestamp = build_crs_dep_timestamp(flight_date, dep_raw)

    cancelled = clean_flag(raw["cancelled_raw"], "CANCELLED")
    diverted = clean_flag(raw["diverted_raw"], "DIVERTED")
    dep_del15 = clean_dep_del15(raw.get("dep_del15_raw"), cancelled)
    is_training_eligible = compute_training_eligibility(dep_del15, cancelled, diverted)

    elapsed_time = parse_elapsed_time(raw.get("crs_elapsed_time_raw"))
    distance = parse_distance(raw.get("distance_raw"))

    return {
        "source_sha256": raw["source_sha256"],
        "source_row_number": int(raw["source_row_number"]),
        "ingestion_run_id": int(raw["ingestion_run_id"]),
        "ingested_at": raw["ingested_at"],
        "flight_date": flight_date,
        "year": year,
        "quarter": quarter,
        "month": month,
        "day_of_month": day_of_month,
        "day_of_week": day_of_week,
        "op_unique_carrier": carrier,
        "origin_airport_id": int(float(raw["origin_airport_id_raw"])),
        "origin": origin,
        "dest_airport_id": int(float(raw["dest_airport_id_raw"])),
        "dest": dest,
        "crs_dep_time_raw": dep_raw,
        "crs_arr_time_raw": arr_raw,
        "crs_dep_time_str": dep_str,
        "crs_arr_time_str": arr_str,
        "crs_dep_time": dep_time,
        "crs_arr_time": arr_time,
        "crs_dep_timestamp": dep_timestamp,
        "crs_elapsed_time": elapsed_time,
        "distance": distance,
        "cancelled": cancelled,
        "diverted": diverted,
        "dep_del15": dep_del15,
        "is_training_eligible": is_training_eligible,
    }


def validate_bronze_partition(conn, checksum: str) -> None:
    """Scan a Bronze partition for malformed values before transformation.

    Surfaces error counts and sample violating rows, failing safely before
    modifying Silver.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (
                    WHERE NOT (
                        flight_date_raw ~ '^\\d{1,2}/\\d{1,2}/\\d{4}'
                        OR flight_date_raw ~ '^\\d{4}-\\d{2}-\\d{2}'
                    )
                ) AS bad_dates,
                COUNT(*) FILTER (
                    WHERE NOT (
                        LPAD(TRIM(crs_dep_time_raw), 4, '0') ~ '^[0-2][0-9][0-5][0-9]$'
                        AND (
                            LPAD(TRIM(crs_dep_time_raw), 4, '0') <= '2359'
                            OR LPAD(TRIM(crs_dep_time_raw), 4, '0') = '2400'
                        )
                    )
                ) AS bad_dep_times,
                COUNT(*) FILTER (
                    WHERE NOT (
                        LPAD(TRIM(crs_arr_time_raw), 4, '0') ~ '^[0-2][0-9][0-5][0-9]$'
                        AND (
                            LPAD(TRIM(crs_arr_time_raw), 4, '0') <= '2359'
                            OR LPAD(TRIM(crs_arr_time_raw), 4, '0') = '2400'
                        )
                    )
                ) AS bad_arr_times,
                COUNT(*) FILTER (
                    WHERE op_unique_carrier_raw IS NULL
                       OR TRIM(op_unique_carrier_raw) = ''
                ) AS bad_carriers,
                COUNT(*) FILTER (
                    WHERE origin_raw IS NULL OR length(TRIM(origin_raw)) != 3
                       OR dest_raw IS NULL OR length(TRIM(dest_raw)) != 3
                ) AS bad_airports,
                COUNT(*) FILTER (
                    WHERE cancelled_raw NOT IN ('0.00', '1.00', '0', '1')
                       OR diverted_raw NOT IN ('0.00', '1.00', '0', '1')
                ) AS bad_flags,
                COUNT(*) FILTER (
                    WHERE dep_del15_raw IS NOT NULL
                      AND TRIM(dep_del15_raw) != ''
                      AND dep_del15_raw NOT IN ('0.00', '1.00', '0', '1')
                ) AS bad_labels
            FROM public.bts_flight_staging
            WHERE source_sha256 = %s
            """,
            (checksum,),
        )
        row = cur.fetchone()
        if row is None or row[0] == 0:
            raise ValueError(f"No Bronze records found for source_sha256={checksum}")

        total, bad_d, bad_dep, bad_arr, bad_c, bad_air, bad_flg, bad_lbl = row
        violations = sum(
            [
                bad_d,
                bad_dep,
                bad_arr,
                bad_c,
                bad_air,
                bad_flg,
                bad_lbl,
            ]
        )
        if violations > 0:
            cur.execute(
                """
                SELECT source_row_number, flight_date_raw, crs_dep_time_raw,
                       op_unique_carrier_raw, origin_raw, dest_raw
                FROM public.bts_flight_staging
                WHERE source_sha256 = %s
                  AND (
                    NOT (
                        flight_date_raw ~ '^\\d{1,2}/\\d{1,2}/\\d{4}'
                        OR flight_date_raw ~ '^\\d{4}-\\d{2}-\\d{2}'
                    )
                    OR NOT (
                        LPAD(TRIM(crs_dep_time_raw), 4, '0')
                        ~ '^[0-2][0-9][0-5][0-9]$'
                    )
                    OR op_unique_carrier_raw IS NULL
                    OR TRIM(op_unique_carrier_raw) = ''
                  )
                LIMIT 5
                """,
                (checksum,),
            )
            samples = cur.fetchall()
            raise ValueError(
                f"Bronze partition {checksum[:12]} failed validation with "
                f"{violations} violations: bad_dates={bad_d}, "
                f"bad_dep_times={bad_dep}, bad_arr_times={bad_arr}, "
                f"bad_carriers={bad_c}, bad_airports={bad_air}, "
                f"bad_flags={bad_flg}, bad_labels={bad_lbl}. "
                f"Sample violating rows: {samples}"
            )


def transform_silver_partition(conn, checksum: str) -> dict:
    """Transform one Bronze partition into Silver idempotently."""
    validate_bronze_partition(conn, checksum)

    columns_sql = ", ".join(TARGET_COLUMNS)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM public.bts_flight_staging
            WHERE source_sha256 = %s
            """,
            (checksum,),
        )
        bronze_count = cur.fetchone()[0]

        cur.execute(
            f"""
            INSERT INTO public.silver_flights ({columns_sql})
            SELECT
                s.source_sha256,
                s.source_row_number,
                s.ingestion_run_id,
                s.ingested_at,
                -- flight_date:
                CASE
                    WHEN POSITION('/' IN s.flight_date_raw) > 0 THEN
                        TO_DATE(
                            split_part(s.flight_date_raw, ' ', 1),
                            'FMMM/FMDD/YYYY'
                        )
                    ELSE
                        SUBSTRING(s.flight_date_raw FROM 1 FOR 10)::DATE
                END AS flight_date,
                s.year_raw::SMALLINT AS year,
                s.quarter_raw::SMALLINT AS quarter,
                s.month_raw::SMALLINT AS month,
                s.day_of_month_raw::SMALLINT AS day_of_month,
                s.day_of_week_raw::SMALLINT AS day_of_week,
                TRIM(s.op_unique_carrier_raw) AS op_unique_carrier,
                s.origin_airport_id_raw::BIGINT AS origin_airport_id,
                TRIM(s.origin_raw) AS origin,
                s.dest_airport_id_raw::BIGINT AS dest_airport_id,
                TRIM(s.dest_raw) AS dest,
                TRIM(s.crs_dep_time_raw) AS crs_dep_time_raw,
                TRIM(s.crs_arr_time_raw) AS crs_arr_time_raw,
                LPAD(TRIM(s.crs_dep_time_raw), 4, '0') AS crs_dep_time_str,
                LPAD(TRIM(s.crs_arr_time_raw), 4, '0') AS crs_arr_time_str,
                -- crs_dep_time (normalized clock time 00:00 to 23:59):
                CASE
                    WHEN LPAD(TRIM(s.crs_dep_time_raw), 4, '0') = '2400'
                        THEN TIME '00:00:00'
                    ELSE TO_TIMESTAMP(
                        LPAD(TRIM(s.crs_dep_time_raw), 4, '0'),
                        'HH24MI'
                    )::TIME
                END AS crs_dep_time,
                -- crs_arr_time (normalized clock time 00:00 to 23:59):
                CASE
                    WHEN LPAD(TRIM(s.crs_arr_time_raw), 4, '0') = '2400'
                        THEN TIME '00:00:00'
                    ELSE TO_TIMESTAMP(
                        LPAD(TRIM(s.crs_arr_time_raw), 4, '0'),
                        'HH24MI'
                    )::TIME
                END AS crs_arr_time,
                -- crs_dep_timestamp:
                CASE
                    WHEN LPAD(TRIM(s.crs_dep_time_raw), 4, '0') = '2400' THEN
                        (CASE
                            WHEN POSITION('/' IN s.flight_date_raw) > 0 THEN
                                TO_DATE(
                                    split_part(s.flight_date_raw, ' ', 1),
                                    'FMMM/FMDD/YYYY'
                                )
                            ELSE
                                SUBSTRING(s.flight_date_raw FROM 1 FOR 10)::DATE
                        END + INTERVAL '1 day')::TIMESTAMP
                    ELSE
                        (CASE
                            WHEN POSITION('/' IN s.flight_date_raw) > 0 THEN
                                TO_DATE(
                                    split_part(s.flight_date_raw, ' ', 1),
                                    'FMMM/FMDD/YYYY'
                                )
                            ELSE
                                SUBSTRING(s.flight_date_raw FROM 1 FOR 10)::DATE
                        END + TO_TIMESTAMP(
                            LPAD(TRIM(s.crs_dep_time_raw), 4, '0'),
                            'HH24MI'
                        )::TIME)::TIMESTAMP
                END AS crs_dep_timestamp,
                NULLIF(
                    TRIM(s.crs_elapsed_time_raw), ''
                )::DOUBLE PRECISION AS crs_elapsed_time,
                TRIM(s.distance_raw)::DOUBLE PRECISION AS distance,
                (s.cancelled_raw::DOUBLE PRECISION = 1.0) AS cancelled,
                (s.diverted_raw::DOUBLE PRECISION = 1.0) AS diverted,
                -- dep_del15: NULL for cancelled flights; otherwise cast:
                CASE
                    WHEN s.cancelled_raw::DOUBLE PRECISION = 1.0 THEN NULL
                    WHEN s.dep_del15_raw IS NULL OR TRIM(s.dep_del15_raw) = '' THEN NULL
                    ELSE s.dep_del15_raw::DOUBLE PRECISION::SMALLINT
                END AS dep_del15,
                -- is_training_eligible:
                (
                    s.cancelled_raw::DOUBLE PRECISION = 0.0
                    AND s.diverted_raw::DOUBLE PRECISION = 0.0
                    AND s.dep_del15_raw IS NOT NULL
                    AND TRIM(s.dep_del15_raw) != ''
                ) AS is_training_eligible
            FROM public.bts_flight_staging s
            WHERE s.source_sha256 = %s
            ON CONFLICT (source_sha256, source_row_number) DO NOTHING
            """,
            (checksum,),
        )
        inserted = cur.rowcount
        skipped = bronze_count - inserted

    return {
        "source_sha256": checksum,
        "records_seen": bronze_count,
        "records_inserted": inserted,
        "records_skipped": skipped,
    }


def get_silver_state(conn, checksum: str) -> tuple[int, int, bool]:
    """Return Bronze count, Silver count, and completeness for an archive."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM public.bts_flight_staging
            WHERE source_sha256 = %s
            """,
            (checksum,),
        )
        bronze_count = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COUNT(*), MIN(source_row_number), MAX(source_row_number)
            FROM public.silver_flights
            WHERE source_sha256 = %s
            """,
            (checksum,),
        )
        silver_count, min_row, max_row = cur.fetchone()

    if bronze_count == 0 or silver_count == 0:
        return bronze_count, silver_count or 0, False

    is_complete = (
        bronze_count == silver_count and min_row == 1 and max_row == bronze_count
    )
    return bronze_count, silver_count, is_complete

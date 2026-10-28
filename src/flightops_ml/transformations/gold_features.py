import logging
import math
from collections.abc import Iterable
from datetime import date, time

LOGGER = logging.getLogger("flightops_ml.transformations.gold_features")

# Split boundary dates
TRAIN_START_DATE = date(2025, 1, 1)
TRAIN_END_DATE = date(2025, 8, 31)
VAL_START_DATE = date(2025, 9, 1)
VAL_END_DATE = date(2025, 10, 31)
TEST_START_DATE = date(2025, 11, 1)
TEST_END_DATE = date(2025, 12, 31)

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_sha256",
    "source_row_number",
    "flight_date",
    "dataset_split",
)

TARGET_COLUMN: str = "dep_del15"

PREDICTOR_ALLOWLIST: tuple[str, ...] = (
    # Temporal & Calendar
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "quarter",
    "is_weekend",
    "dep_hour",
    "dep_minute",
    "sin_dep_time",
    "cos_dep_time",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_month",
    "cos_month",
    # Carrier & Route Itinerary
    "op_unique_carrier",
    "origin_airport_id",
    "dest_airport_id",
    "distance",
    "crs_elapsed_time_clean",
    # Verified Historical Spatial Coordinates
    "origin_latitude",
    "origin_longitude",
    "dest_latitude",
    "dest_longitude",
)

ALL_GOLD_COLUMNS: tuple[str, ...] = (
    *PROVENANCE_COLUMNS,
    TARGET_COLUMN,
    *PREDICTOR_ALLOWLIST,
)

FORBIDDEN_LEAKAGE_TERMS: tuple[str, ...] = (
    "actual",
    "arr_time",
    "dep_time_raw",
    "arr_delay",
    "dep_delay",
    "taxi",
    "air_time",
    "wheels",
    "cancel",
    "divert",
    "weather",
    "carrier_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "tz_offset",
    "variation",
    "impute",
)


def validate_feature_allowlist(feature_names: Iterable[str]) -> None:
    """Enforce strict feature allowlist and leakage prohibition.

    Raises:
        ValueError: If target column appears in feature list, if an unallowed
                    feature is present, or if any blacklisted substring appears.
    """
    feature_set = set(feature_names)

    if TARGET_COLUMN in feature_set:
        raise ValueError(
            f"Target leakage detected: {TARGET_COLUMN!r} found in predictor feature set"
        )

    for col in feature_set:
        col_lower = col.lower()
        for term in FORBIDDEN_LEAKAGE_TERMS:
            if term in col_lower:
                raise ValueError(
                    f"Forbidden leakage term {term!r} detected in column {col!r}"
                )

    unallowed = feature_set - set(PREDICTOR_ALLOWLIST)
    if unallowed:
        raise ValueError(
            f"Unallowed features detected in Gold feature matrix: {sorted(unallowed)}"
        )


def assign_dataset_split(flight_date: date) -> str:
    """Deterministically assign chronological dataset split based on flight date.

    Splits:
        - train: 2025-01-01 through 2025-08-31
        - validation: 2025-09-01 through 2025-10-31
        - test: 2025-11-01 through 2025-12-31
    """
    if flight_date < TRAIN_START_DATE:
        raise ValueError(
            f"Flight date precedes 2025 train boundary: "
            f"{flight_date} < {TRAIN_START_DATE}"
        )
    if flight_date <= TRAIN_END_DATE:
        return "train"
    if flight_date <= VAL_END_DATE:
        return "validation"
    if flight_date <= TEST_END_DATE:
        return "test"
    raise ValueError(
        f"Flight date exceeds 2025 test boundary: {flight_date} > {TEST_END_DATE}"
    )


def compute_cyclical_dep_time(
    dep_time_val: time | str,
) -> tuple[int, int, float, float]:
    """Extract hour, minute, and continuous cyclical representations of departure time.

    Consistent with the project 2400 convention:
    2400 string or 00:00:00 clock time maps to dep_hour=0, dep_minute=0.
    """
    if isinstance(dep_time_val, str):
        val = dep_time_val.strip()
        if val in ("2400", "24:00", "24:00:00"):
            hour, minute = 0, 0
        else:
            cleaned = val.replace(":", "")
            padded = cleaned.zfill(4)
            hour = int(padded[:2])
            minute = int(padded[2:4])
    elif isinstance(dep_time_val, time):
        hour = dep_time_val.hour
        minute = dep_time_val.minute
    else:
        raise TypeError(f"Unsupported dep_time type: {type(dep_time_val)}")

    if hour == 24 and minute == 0:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid hour/minute: hour={hour}, minute={minute}")

    total_minutes = hour * 60 + minute
    angle = 2.0 * math.pi * (total_minutes / 1440.0)
    sin_time = round(math.sin(angle), 6)
    cos_time = round(math.cos(angle), 6)

    return hour, minute, sin_time, cos_time


def compute_cyclical_day_of_week(day_of_week: int) -> tuple[float, float]:
    """Compute continuous cyclical representation of ISO day of week (1=Mon..7=Sun)."""
    if not 1 <= day_of_week <= 7:
        raise ValueError(f"day_of_week must be in 1..7, got {day_of_week}")

    angle = 2.0 * math.pi * ((day_of_week - 1) / 7.0)
    return round(math.sin(angle), 6), round(math.cos(angle), 6)


def compute_cyclical_month(month: int) -> tuple[float, float]:
    """Compute continuous cyclical representation of calendar month (1..12)."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")

    angle = 2.0 * math.pi * ((month - 1) / 12.0)
    return round(math.sin(angle), 6), round(math.cos(angle), 6)


def transform_gold_row(
    silver_row: dict,
    orig_coords: tuple[float, float],
    dest_coords: tuple[float, float],
) -> dict:
    """Validate and transform a single Silver flight record into Gold features.

    Excludes records with invalid, missing, or nonpositive scheduled duration.
    """
    if not silver_row.get("is_training_eligible", True):
        raise ValueError("Attempted to transform an ineligible flight into Gold")

    elapsed = silver_row.get("crs_elapsed_time")
    if elapsed is None or elapsed <= 0:
        raise ValueError(
            f"Invalid scheduled duration for Gold: crs_elapsed_time={elapsed!r}"
        )

    f_date = silver_row["flight_date"]
    if isinstance(f_date, str):
        f_date = date.fromisoformat(f_date)

    split = assign_dataset_split(f_date)
    target = int(silver_row["dep_del15"])
    if target not in (0, 1):
        raise ValueError(f"Invalid dep_del15 target label: {target!r}")

    dep_time = silver_row["crs_dep_time"]
    hour, minute, sin_time, cos_time = compute_cyclical_dep_time(dep_time)

    month = int(silver_row["month"])
    day_of_week = int(silver_row["day_of_week"])
    sin_dow, cos_dow = compute_cyclical_day_of_week(day_of_week)
    sin_mon, cos_mon = compute_cyclical_month(month)
    is_weekend = day_of_week in (6, 7)

    orig_id = int(silver_row["origin_airport_id"])
    dest_id = int(silver_row["dest_airport_id"])

    orig_lat, orig_lon = orig_coords
    dest_lat, dest_lon = dest_coords
    if not (-90 <= orig_lat <= 90 and -180 <= orig_lon <= 180):
        raise ValueError(f"Invalid origin coordinates: lat={orig_lat}, lon={orig_lon}")
    if not (-90 <= dest_lat <= 90 and -180 <= dest_lon <= 180):
        raise ValueError(f"Invalid dest coordinates: lat={dest_lat}, lon={dest_lon}")

    row = {
        "source_sha256": silver_row["source_sha256"],
        "source_row_number": int(silver_row["source_row_number"]),
        "flight_date": f_date,
        "dataset_split": split,
        "dep_del15": target,
        "year": int(silver_row["year"]),
        "month": month,
        "day_of_month": int(silver_row["day_of_month"]),
        "day_of_week": day_of_week,
        "quarter": int(silver_row["quarter"]),
        "is_weekend": is_weekend,
        "dep_hour": hour,
        "dep_minute": minute,
        "sin_dep_time": sin_time,
        "cos_dep_time": cos_time,
        "sin_day_of_week": sin_dow,
        "cos_day_of_week": cos_dow,
        "sin_month": sin_mon,
        "cos_month": cos_mon,
        "op_unique_carrier": str(silver_row["op_unique_carrier"]).strip(),
        "origin_airport_id": orig_id,
        "dest_airport_id": dest_id,
        "distance": float(silver_row["distance"]),
        "crs_elapsed_time_clean": float(elapsed),
        "origin_latitude": float(orig_lat),
        "origin_longitude": float(orig_lon),
        "dest_latitude": float(dest_lat),
        "dest_longitude": float(dest_lon),
    }

    validate_feature_allowlist(
        [col for col in row if col not in (*PROVENANCE_COLUMNS, TARGET_COLUMN)]
    )
    return row


AIRPORT_POINT_IN_TIME_JOIN_SQL: str = """
    LEFT JOIN public.airport_reference orig
        ON s.origin_airport_id = orig.airport_id
       AND s.flight_date >= orig.airport_start_date
       AND (orig.airport_thru_date IS NULL
            OR s.flight_date <= orig.airport_thru_date)
    LEFT JOIN public.airport_reference dest
        ON s.dest_airport_id = dest.airport_id
       AND s.flight_date >= dest.airport_start_date
       AND (dest.airport_thru_date IS NULL
            OR s.flight_date <= dest.airport_thru_date)
"""

RECONCILIATION_QUERIES: dict[str, str] = {
    "row_counts_by_split": """
        SELECT dataset_split, COUNT(*) AS gold_rows,
               COUNT(*) FILTER (WHERE dep_del15 = 1) AS delayed_rows,
               ROUND(
                   COUNT(*) FILTER (WHERE dep_del15 = 1)::NUMERIC
                   / COUNT(*) * 100, 2
               ) AS delay_rate
        FROM public.gold_features
        GROUP BY dataset_split
        ORDER BY MIN(flight_date);
    """,
    "silver_eligible_vs_gold_counts": """
        WITH silver_counts AS (
            SELECT
                CASE
                    WHEN flight_date < DATE '2025-09-01' THEN 'train'
                    WHEN flight_date < DATE '2025-11-01' THEN 'validation'
                    ELSE 'test'
                END AS dataset_split,
                COUNT(*) AS eligible_silver_rows,
                COUNT(*) FILTER (
                    WHERE crs_elapsed_time IS NULL
                ) AS missing_duration_rows,
                COUNT(*) FILTER (
                    WHERE crs_elapsed_time <= 0
                ) AS nonpositive_duration_rows,
                COUNT(*) FILTER (
                    WHERE crs_elapsed_time IS NOT NULL AND crs_elapsed_time > 0
                ) AS expected_gold_rows
            FROM public.silver_flights
            WHERE is_training_eligible = TRUE
            GROUP BY 1
        ),
        gold_counts AS (
            SELECT dataset_split, COUNT(*) AS actual_gold_rows
            FROM public.gold_features
            GROUP BY dataset_split
        )
        SELECT
            s.dataset_split,
            s.eligible_silver_rows,
            s.missing_duration_rows,
            s.nonpositive_duration_rows,
            s.expected_gold_rows,
            COALESCE(g.actual_gold_rows, 0) AS actual_gold_rows,
            (s.expected_gold_rows = COALESCE(g.actual_gold_rows, 0)) AS is_reconciled
        FROM silver_counts s
        LEFT JOIN gold_counts g ON s.dataset_split = g.dataset_split
        ORDER BY s.dataset_split;
    """,
    "missing_coordinates_check": """
        SELECT
            COUNT(*) FILTER (
                WHERE origin_latitude IS NULL OR origin_longitude IS NULL
            ) AS missing_origin_coords,
            COUNT(*) FILTER (
                WHERE dest_latitude IS NULL OR dest_longitude IS NULL
            ) AS missing_dest_coords
        FROM public.gold_features;
    """,
    "nonpositive_elapsed_time_check": """
        SELECT COUNT(*) AS nonpositive_elapsed_rows
        FROM public.gold_features
        WHERE crs_elapsed_time_clean <= 0;
    """,
    "duplicate_source_rows_check": """
        SELECT source_sha256, source_row_number, COUNT(*)
        FROM public.gold_features
        GROUP BY source_sha256, source_row_number
        HAVING COUNT(*) > 1;
    """,
}

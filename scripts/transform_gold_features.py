import argparse
import logging
import sys

from flightops_ml.ingestion.loader import database_connection
from flightops_ml.transformations.gold_features import (
    ALL_GOLD_COLUMNS,
    RECONCILIATION_QUERIES,
)

LOGGER = logging.getLogger("flightops_ml.transform_gold_features")


def check_gold_table_exists(conn) -> bool:
    """Return True if public.gold_features table exists in the database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'gold_features'
            )
            """
        )
        return bool(cur.fetchone()[0])


def audit_silver_and_exclusions(conn) -> list[dict]:
    """Audit Silver eligible counts, exclusions, and expected Gold rows."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                CASE
                    WHEN flight_date < DATE '2025-09-01' THEN 'train'
                    WHEN flight_date < DATE '2025-11-01' THEN 'validation'
                    ELSE 'test'
                END AS dataset_split,
                MIN(flight_date) AS min_date,
                MAX(flight_date) AS max_date,
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
            ORDER BY MIN(flight_date);
            """
        )
        return [
            {
                "dataset_split": row[0],
                "min_date": str(row[1]),
                "max_date": str(row[2]),
                "eligible_silver_rows": row[3],
                "missing_duration_rows": row[4],
                "nonpositive_duration_rows": row[5],
                "total_excluded_rows": row[4] + row[5],
                "expected_gold_rows": row[6],
            }
            for row in cur.fetchall()
        ]


def get_actual_gold_counts(conn) -> dict[str, int]:
    """Retrieve actual Gold row counts by split if table exists."""
    if not check_gold_table_exists(conn):
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT dataset_split, COUNT(*)
            FROM public.gold_features
            GROUP BY dataset_split;
            """
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def run_dry_run(conn) -> None:
    """Execute read-only dry run audit.

    Does NOT write any database records or modify any schema objects.
    """
    LOGGER.info("=== STARTING READ-ONLY DRY RUN (NO DATABASE WRITES) ===")

    table_exists = check_gold_table_exists(conn)
    LOGGER.info("Target table public.gold_features exists: %s", table_exists)
    if not table_exists:
        LOGGER.info(
            "NOTE: Migration 006 has not been applied yet. "
            "Apply sql/006_create_gold_features.sql prior to --execute."
        )

    # 1. Audit Silver eligible population and explicit exclusions
    audit_rows = audit_silver_and_exclusions(conn)
    actual_gold = get_actual_gold_counts(conn)

    total_eligible = sum(r["eligible_silver_rows"] for r in audit_rows)
    total_missing = sum(r["missing_duration_rows"] for r in audit_rows)
    total_nonpositive = sum(r["nonpositive_duration_rows"] for r in audit_rows)
    total_excluded = sum(r["total_excluded_rows"] for r in audit_rows)
    total_expected = sum(r["expected_gold_rows"] for r in audit_rows)

    LOGGER.info("--- SPLIT RECONCILIATION & EXCLUSION REPORT ---")
    LOGGER.info(
        "%-12s | %-15s | %-16s | %-18s | %-14s | %-18s | %-16s",
        "Split",
        "Eligible Silver",
        "Missing Duration",
        "Nonpositive Dur.",
        "Total Excluded",
        "Expected Gold Rows",
        "Actual Gold Rows",
    )
    LOGGER.info("-" * 115)
    for r in audit_rows:
        split_name = r["dataset_split"]
        actual_count = actual_gold.get(split_name, 0)
        LOGGER.info(
            "%-12s | %15s | %16s | %18s | %14s | %18s | %16s",
            split_name,
            f"{r['eligible_silver_rows']:,}",
            f"{r['missing_duration_rows']:,}",
            f"{r['nonpositive_duration_rows']:,}",
            f"{r['total_excluded_rows']:,}",
            f"{r['expected_gold_rows']:,}",
            f"{actual_count:,}" if table_exists else "N/A (pending 006)",
        )
    LOGGER.info("-" * 115)
    LOGGER.info(
        "%-12s | %15s | %16s | %18s | %14s | %18s | %16s",
        "TOTAL",
        f"{total_eligible:,}",
        f"{total_missing:,}",
        f"{total_nonpositive:,}",
        f"{total_excluded:,}",
        f"{total_expected:,}",
        f"{sum(actual_gold.values()):,}" if table_exists else "N/A (pending 006)",
    )

    # 2. Verify airport reference join coverage on eligible, positive-duration flights
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_filtered,
                COUNT(orig.airport_id) AS matched_origin,
                COUNT(dest.airport_id) AS matched_dest
            FROM public.silver_flights s
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
            WHERE s.is_training_eligible = TRUE
              AND s.crs_elapsed_time IS NOT NULL
              AND s.crs_elapsed_time > 0;
            """
        )
        tot, m_orig, m_dest = cur.fetchone()
        LOGGER.info(
            "Airport join verification on filtered records: "
            "eligible_valid=%s, origin_match=%s, dest_match=%s",
            f"{tot:,}",
            f"{m_orig:,}",
            f"{m_dest:,}",
        )

    LOGGER.info(
        "=== DRY RUN COMPLETE: 0 rows written. "
        "Would transform %s rows into public.gold_features "
        "(excluded %s invalid-duration rows). ===",
        f"{total_expected:,}",
        f"{total_excluded:,}",
    )


def execute_transformation(conn, replace: bool = False) -> int:
    """Execute Gold V1 batch transformation inside a database transaction."""
    if not check_gold_table_exists(conn):
        LOGGER.error(
            "Target table public.gold_features does not exist! "
            "Apply migration sql/006_create_gold_features.sql first."
        )
        return 1

    with conn.cursor() as cur:
        # Check existing table state to ensure safe, idempotent rerun behavior
        cur.execute("SELECT COUNT(*) FROM public.gold_features;")
        existing_count = cur.fetchone()[0]

        if existing_count > 0 and not replace:
            cur.execute(RECONCILIATION_QUERIES["silver_eligible_vs_gold_counts"])
            reconciliation_results = cur.fetchall()
            if reconciliation_results and all(row[6] for row in reconciliation_results):
                LOGGER.info(
                    "Gold table public.gold_features is already complete and "
                    "reconciled (%s rows). Re-execution skipped. "
                    "Pass --replace to rebuild.",
                    f"{existing_count:,}",
                )
                return 0
            LOGGER.warning(
                "Gold table contains %s rows and is not fully reconciled. "
                "Resuming insertion with ON CONFLICT DO NOTHING...",
                f"{existing_count:,}",
            )

        if replace:
            LOGGER.info("Flag --replace specified: truncating public.gold_features...")
            cur.execute("TRUNCATE TABLE public.gold_features;")

        columns_sql = ", ".join(ALL_GOLD_COLUMNS)
        LOGGER.info("Inserting transformed rows into public.gold_features...")

        cur.execute(
            f"""
            INSERT INTO public.gold_features ({columns_sql})
            SELECT
                s.source_sha256,
                s.source_row_number,
                s.flight_date,
                -- dataset_split
                CASE
                    WHEN s.flight_date < DATE '2025-09-01' THEN 'train'
                    WHEN s.flight_date < DATE '2025-11-01' THEN 'validation'
                    ELSE 'test'
                END AS dataset_split,
                s.dep_del15,
                s.year,
                s.month,
                s.day_of_month,
                s.day_of_week,
                s.quarter,
                (s.day_of_week IN (6, 7)) AS is_weekend,
                -- dep_hour and dep_minute
                EXTRACT(HOUR FROM s.crs_dep_time)::SMALLINT AS dep_hour,
                EXTRACT(MINUTE FROM s.crs_dep_time)::SMALLINT AS dep_minute,
                -- cyclical departure time
                ROUND(
                    SIN(2.0 * PI() * (
                        EXTRACT(HOUR FROM s.crs_dep_time) * 60
                        + EXTRACT(MINUTE FROM s.crs_dep_time)
                    ) / 1440.0)::NUMERIC, 6
                )::DOUBLE PRECISION AS sin_dep_time,
                ROUND(
                    COS(2.0 * PI() * (
                        EXTRACT(HOUR FROM s.crs_dep_time) * 60
                        + EXTRACT(MINUTE FROM s.crs_dep_time)
                    ) / 1440.0)::NUMERIC, 6
                )::DOUBLE PRECISION AS cos_dep_time,
                -- cyclical day of week
                ROUND(
                    SIN(2.0 * PI() * (s.day_of_week - 1) / 7.0)::NUMERIC, 6
                )::DOUBLE PRECISION AS sin_day_of_week,
                ROUND(
                    COS(2.0 * PI() * (s.day_of_week - 1) / 7.0)::NUMERIC, 6
                )::DOUBLE PRECISION AS cos_day_of_week,
                -- cyclical month
                ROUND(
                    SIN(2.0 * PI() * (s.month - 1) / 12.0)::NUMERIC, 6
                )::DOUBLE PRECISION AS sin_month,
                ROUND(
                    COS(2.0 * PI() * (s.month - 1) / 12.0)::NUMERIC, 6
                )::DOUBLE PRECISION AS cos_month,
                -- carrier and route
                s.op_unique_carrier,
                s.origin_airport_id,
                s.dest_airport_id,
                s.distance,
                -- crs_elapsed_time_clean (from positive Silver value)
                s.crs_elapsed_time AS crs_elapsed_time_clean,
                -- origin and dest coordinates
                orig.latitude AS origin_latitude,
                orig.longitude AS origin_longitude,
                dest.latitude AS dest_latitude,
                dest.longitude AS dest_longitude
            FROM public.silver_flights s
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
            WHERE s.is_training_eligible = TRUE
              AND s.crs_elapsed_time IS NOT NULL
              AND s.crs_elapsed_time > 0
            ON CONFLICT (source_sha256, source_row_number, dataset_split) DO NOTHING;
            """
        )
        inserted_rows = cur.rowcount
        LOGGER.info("Inserted %s rows into public.gold_features", f"{inserted_rows:,}")

        # Post-transformation audit & reconciliation
        cur.execute(RECONCILIATION_QUERIES["silver_eligible_vs_gold_counts"])
        reconciliation_results = cur.fetchall()
        LOGGER.info(
            "Gold reconciliation results (split, eligible_silver, "
            "missing_dur, nonpos_dur, expected, actual, reconciled):"
        )
        all_reconciled = True
        for row in reconciliation_results:
            LOGGER.info("  %s", row)
            if not row[6]:
                all_reconciled = False

        if not all_reconciled:
            raise RuntimeError(
                "Gold reconciliation failed: row counts do not match "
                "expected Silver eligible minus exclusions!"
            )

    conn.commit()
    LOGGER.info("SUCCESS: Gold V1 transformation committed and reconciled.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transform eligible Silver flights into Gold V1 feature dataset."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute transformation against database. Default is read-only dry run.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Truncate public.gold_features before inserting if rebuilding from scratch."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with database_connection() as conn:
        if not args.execute:
            run_dry_run(conn)
            return 0
        return execute_transformation(conn, replace=args.replace)


if __name__ == "__main__":
    sys.exit(main())

import argparse
import logging
import sys

from flightops_ml.ingestion.loader import database_connection
from flightops_ml.transformations.silver_flights import (
    get_silver_state,
    transform_silver_partition,
)

LOGGER = logging.getLogger("flightops_ml.transform_silver_flights")


def get_target_archives(
    conn,
    year: int,
    start_month: int,
    end_month: int,
) -> list[tuple[int, int, str, str]]:
    """Return successfully ingested BTS archives for the requested year and months."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (source_year, source_month)
                source_year,
                source_month,
                source_sha256,
                source_filename
            FROM public.ingestion_runs
            WHERE source_name = 'BTS'
              AND source_year = %s
              AND source_month BETWEEN %s AND %s
              AND status = 'succeeded'
            ORDER BY source_year, source_month, run_id DESC
            """,
            (year, start_month, end_month),
        )
        return cur.fetchall()


def process_archive(
    conn,
    year: int,
    month: int,
    checksum: str,
    filename: str,
    execute: bool,
) -> str:
    """Validate, dry-run or transform a single monthly archive into Silver."""
    bronze_rows, silver_rows, is_complete = get_silver_state(conn, checksum)

    if is_complete:
        LOGGER.info(
            "%04d-%02d (%s): SKIP — already complete in Silver (%s rows)",
            year,
            month,
            filename,
            f"{silver_rows:,}",
        )
        return "skipped"

    if not execute:
        LOGGER.info(
            "%04d-%02d (%s): WOULD TRANSFORM — "
            "Bronze rows: %s, existing Silver rows: %s",
            year,
            month,
            filename,
            f"{bronze_rows:,}",
            f"{silver_rows:,}",
        )
        return "planned"

    LOGGER.info(
        "%04d-%02d (%s): transforming %s Bronze rows into Silver",
        year,
        month,
        filename,
        f"{bronze_rows:,}",
    )
    result = transform_silver_partition(conn, checksum)
    conn.commit()

    _, silver_after, complete_after = get_silver_state(conn, checksum)
    if not complete_after:
        raise RuntimeError(
            f"{year}-{month:02d} ({filename}): post-transformation check failed; "
            f"expected={result['records_seen']}, silver_rows={silver_after}"
        )

    LOGGER.info(
        "%04d-%02d (%s): SUCCESS — inserted=%s, skipped=%s",
        year,
        month,
        filename,
        f"{result['records_inserted']:,}",
        f"{result['records_skipped']:,}",
    )
    return "transformed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transform Bronze BTS staging data into typed Silver flights."
    )
    parser.add_argument("--year", type=int, default=2025, help="Target year")
    parser.add_argument(
        "--start-month", type=int, default=1, help="Starting month (1-12)"
    )
    parser.add_argument("--end-month", type=int, default=12, help="Ending month (1-12)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the transformation. Default is dry run.",
    )
    args = parser.parse_args(argv)

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
        "%s: transforming %s months %02d through %02d into public.silver_flights",
        mode,
        args.year,
        args.start_month,
        args.end_month,
    )

    results = {"transformed": 0, "skipped": 0, "planned": 0, "failed": 0}

    with database_connection() as conn:
        archives = get_target_archives(
            conn, args.year, args.start_month, args.end_month
        )

        if not archives:
            LOGGER.warning(
                "No successful BTS ingestion runs found for year %s months %02d-%02d",
                args.year,
                args.start_month,
                args.end_month,
            )
            return 0

        for year, month, checksum, filename in archives:
            try:
                status = process_archive(
                    conn, year, month, checksum, filename, args.execute
                )
                results[status] += 1
            except Exception:
                conn.rollback()
                results["failed"] += 1
                LOGGER.exception(
                    "Transformation failed for %04d-%02d (%s)",
                    year,
                    month,
                    filename,
                )

    LOGGER.info("FINAL SUMMARY: %s", results)
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI script to orchestrate bulk weather observation ingestion for FlightOps ML."""

import argparse
import csv
import logging
import sys
from pathlib import Path

from flightops_ml.ingestion.weather_auditor import REPRESENTATIVE_PILOT_STATIONS
from flightops_ml.ingestion.weather_bulk_ingester import execute_weather_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("ingest_weather_year")


def load_target_stations(
    mapping_path: Path, stids: list[str] | None = None
) -> list[str]:
    """Load station IDs from mapping CSV, optionally filtering by target stids."""
    with mapping_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_stids = [
            r["station_id"].strip().upper() for r in reader if r.get("station_id")
        ]

    unique_stids = sorted(set(all_stids))
    if stids is not None:
        target_set = set(s.strip().upper() for s in stids)
        return [s for s in unique_stids if s in target_set]

    return unique_stids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest 2025 IEM ASOS weather observations into "
            "public.weather_observations."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--top-50",
        action="store_true",
        help="Ingest the Top 50 origin airport weather stations for 2025.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest all 352 mapped airport weather stations.",
    )
    group.add_argument(
        "--pilot",
        action="store_true",
        help="Ingest the 40 representative pilot stations.",
    )
    group.add_argument(
        "--stations",
        type=str,
        help="Comma-separated list of station IDs to ingest.",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Calendar year to ingest (default: 2025)",
    )
    parser.add_argument(
        "--months",
        type=str,
        default=None,
        help="Comma-separated list of months (1-12) to ingest (default: all 12)",
    )
    parser.add_argument(
        "--annual",
        action="store_true",
        default=True,
        help="Fetch full calendar year in single queries (default: True)",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Force monthly queries instead of annual",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of stations per request chunk (default: 5)",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.5,
        help="Polite pause between HTTP requests (default: 1.5s)",
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="data/processed/weather/airport_station_mapping.csv",
        help="Path to airport_station_mapping.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/weather/2025",
        help="Directory to save raw batch CSV files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse without staging rows into PostgreSQL.",
    )

    args = parser.parse_args()

    mapping_path = Path(args.mapping_path)
    if not mapping_path.exists():
        LOGGER.error("Mapping file not found at: %s", mapping_path)
        sys.exit(1)

    if args.top_50:
        top_50_csv = Path("data/processed/weather/top_50_airports.csv")
        if not top_50_csv.exists():
            LOGGER.error("Top 50 CSV not found at: %s", top_50_csv)
            sys.exit(1)
        with top_50_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            target_stids = [
                r["station_id"].strip().upper() for r in reader if r.get("station_id")
            ]
        scope_name = "TOP 50 (50 busiest hub stations)"
        station_list = sorted(set(target_stids))
    elif args.pilot:
        target_stids = REPRESENTATIVE_PILOT_STATIONS
        scope_name = "PILOT (40 stations)"
        station_list = load_target_stations(mapping_path, target_stids)
    elif args.stations:
        target_stids = [
            s.strip().upper() for s in args.stations.split(",") if s.strip()
        ]
        scope_name = f"CUSTOM ({len(target_stids)} stations)"
        station_list = load_target_stations(mapping_path, target_stids)
    else:
        target_stids = None
        scope_name = "ALL (352 stations)"
        station_list = load_target_stations(mapping_path, target_stids)

    if not station_list:
        LOGGER.error("No valid stations selected from mapping.")
        sys.exit(1)

    is_annual = not args.monthly and args.annual

    target_months = None
    if args.months:
        target_months = [int(m.strip()) for m in args.months.split(",") if m.strip()]
        is_annual = False

    raw_dir = Path(args.raw_dir)
    manifest_path = raw_dir / ".ingestion_manifest.json"

    months_label = target_months or ("All 12 Months" if not is_annual else "Full 2025")

    print("==================================================================")
    print("FlightOps ML — Bulk Weather Observation Ingestion (2025)")
    print(f"Scope: {scope_name}")
    print(f"Target Stations: {len(station_list)}")
    print(f"Mode: {'Annual (Full Year)' if is_annual else 'Monthly'}")
    print(f"Target Months: {months_label}")
    print(f"Chunk Size: {args.chunk_size} stations/chunk")
    print(f"Polite Pause: {args.pause_seconds}s")
    print(f"Raw Output Directory: {raw_dir}")
    print(f"Dry Run: {args.dry_run}")
    print("==================================================================")

    results = execute_weather_ingestion(
        station_ids=station_list,
        year=args.year,
        months=target_months,
        annual=is_annual,
        chunk_size=args.chunk_size,
        pause_seconds=args.pause_seconds,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )

    total_seen = sum(r.records_seen for r in results)
    total_ins = sum(r.records_inserted for r in results)
    total_skip = sum(r.records_skipped for r in results)

    print("\n==================================================================")
    print("Ingestion Execution Finished")
    print(f"Batches Processed: {len(results)}")
    print(f"Records Seen: {total_seen:,}")
    print(f"Records Inserted: {total_ins:,}")
    print(f"Records Skipped/Conflicts: {total_skip:,}")
    print("==================================================================")


if __name__ == "__main__":
    main()

"""CLI script to export airport inventory and map origin airports to
IEM weather stations.
"""

import argparse
from pathlib import Path

from flightops_ml.ingestion.weather_station_mapper import (
    export_airport_inventory,
    map_airports_to_weather_stations,
)


def find_catalogue_file(specified_path: str | None = None) -> Path:
    if specified_path:
        p = Path(specified_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified catalogue path not found: {specified_path}")

    candidates = [
        Path("data/raw/weather/weather-station-catalogue.csv"),
        Path("data/weather-station-catalogue.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Weather station catalogue not found. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map FlightOps ML gold origin airports to IEM weather stations."
    )
    parser.add_argument(
        "--export-inventory",
        action="store_true",
        default=False,
        help="Query PostgreSQL and export data/raw/weather/airport_inventory.csv",
    )
    parser.add_argument(
        "--inventory-path",
        type=str,
        default="data/raw/weather/airport_inventory.csv",
        help="Path to the airport inventory CSV",
    )
    parser.add_argument(
        "--catalogue-path",
        type=str,
        default=None,
        help="Path to the weather station catalogue CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/weather",
        help="Directory to save mapping outputs",
    )
    parser.add_argument(
        "--master-cord-path",
        type=str,
        default="data/raw/airport_reference/T_MASTER_CORD.csv",
        help="Path to T_MASTER_CORD.csv for offline metadata lookup",
    )

    args = parser.parse_args()

    inventory_path = Path(args.inventory_path)
    catalogue_path = find_catalogue_file(args.catalogue_path)
    output_dir = Path(args.output_dir)
    master_cord_path = Path(args.master_cord_path) if args.master_cord_path else None

    print("=== FlightOps ML Weather Station Mapping ===")
    print(f"Catalogue path: {catalogue_path}")

    # Check if inventory export is requested or file is missing
    if args.export_inventory or not inventory_path.exists():
        print(
            "Exporting airport inventory from public.gold_features to "
            f"{inventory_path}..."
        )
        inv_res = export_airport_inventory(inventory_path)
        print(
            f"Inventory export complete: {inv_res['record_count']} airports, "
            f"{inv_res['total_flights']:,} flights, SHA256={inv_res['sha256'][:16]}..."
        )
    else:
        print(f"Using existing airport inventory at: {inventory_path}")

    print("Executing airport-to-station mapping...")
    summary = map_airports_to_weather_stations(
        inventory_path=inventory_path,
        catalogue_path=catalogue_path,
        output_dir=output_dir,
        master_cord_path=master_cord_path,
    )

    print("\n=== Mapping Execution Summary ===")
    print(f"Airports Mapped: {summary['total_airports_mapped']}")
    print(f"Match Types: {summary['match_type_counts']}")
    print(
        f"Distance Min/Median/Max/Mean: {summary['distance_summary_km']['min']} / "
        f"{summary['distance_summary_km']['median']} / "
        f"{summary['distance_summary_km']['max']} / "
        f"{summary['distance_summary_km']['mean']} km"
    )
    print(
        "Airports within 5 km of station: "
        f"{summary['on_airport_count_le_5km']} / {summary['total_airports_mapped']}"
    )
    print(
        "Stations with active 2025 archive: "
        f"{summary['active_2025_archive_count']} / {summary['total_airports_mapped']}"
    )
    print(f"Airports flagged for manual review: {summary['flagged_review_count']}")
    print(f"\nOutputs generated in {output_dir}:")
    for f in summary["files_generated"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

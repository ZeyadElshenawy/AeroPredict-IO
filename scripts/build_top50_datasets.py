"""Build and verify Top 50 Hubs modeling datasets (Baseline vs. Weather).

Applies migration 007, populates public.top_50_airports from the reference CSV,
and reconciles row counts, target rates, and weather join metrics across
Train, Validation, and Test splits.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

from flightops_ml.ingestion.loader import database_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("build_top50_datasets")


def apply_migration(sql_path: Path) -> None:
    """Read and execute SQL migration file."""
    LOGGER.info("Applying migration from: %s", sql_path)
    sql_text = sql_path.read_text(encoding="utf-8")
    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()
    LOGGER.info("Migration applied successfully.")


def populate_top_50_airports(csv_path: Path) -> int:
    """Upsert top 50 airport reference data into public.top_50_airports."""
    LOGGER.info("Loading reference airports from: %s", csv_path)
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                int(r["origin_airport_id"]),
                r["airport_code"].strip().upper(),
                r["station_id"].strip().upper(),
                r["station_name"].strip(),
                r["network"].strip(),
                r["iana_timezone"].strip(),
                int(r["train_flights"]),
                float(r["pct_of_train"]),
                float(r["distance_km"]),
            )
            for r in reader
        ]

    upsert_sql = """
        INSERT INTO public.top_50_airports (
            origin_airport_id, airport_code, station_id, station_name,
            network, iana_timezone, train_flights, pct_of_train, distance_km
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (origin_airport_id) DO UPDATE SET
            airport_code = EXCLUDED.airport_code,
            station_id = EXCLUDED.station_id,
            station_name = EXCLUDED.station_name,
            network = EXCLUDED.network,
            iana_timezone = EXCLUDED.iana_timezone,
            train_flights = EXCLUDED.train_flights,
            pct_of_train = EXCLUDED.pct_of_train,
            distance_km = EXCLUDED.distance_km;
    """

    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(upsert_sql, rows)
        conn.commit()

    LOGGER.info("Populated %d airports into public.top_50_airports.", len(rows))
    return len(rows)


def verify_datasets() -> dict[str, dict[str, int | float]]:
    """Reconcile split counts and delay rates between Baseline and Weather views."""
    LOGGER.info("Reconciling Baseline and Weather datasets by partition split...")

    results: dict[str, dict[str, int | float]] = {}

    with database_connection() as conn:
        with conn.cursor() as cur:
            # 1. Baseline view counts
            cur.execute("""
                SELECT
                    dataset_split,
                    COUNT(*) as total_rows,
                    SUM(dep_del15) as delayed_rows,
                    ROUND(AVG(dep_del15) * 100, 2) as delay_rate_pct
                FROM public.vw_gold_top50_baseline
                GROUP BY dataset_split
                ORDER BY CASE dataset_split
                    WHEN 'train' THEN 1
                    WHEN 'validation' THEN 2
                    WHEN 'test' THEN 3
                END;
            """)
            baseline_rows = cur.fetchall()

            for split, total, delayed, rate in baseline_rows:
                results[split] = {
                    "baseline_total": total,
                    "baseline_delayed": delayed,
                    "baseline_rate_pct": float(rate),
                }

    return results


def verify_weather_join_metrics(sample_limit: int = 100_000) -> dict[str, float]:
    """Sample weather view to calculate match rate, freshness, and missingness."""
    LOGGER.info(
        "Auditing weather join metrics on %s sample flights...", f"{sample_limit:,}"
    )

    query = f"""
        WITH sample AS (
            SELECT *
            FROM public.vw_gold_top50_weather
            LIMIT {sample_limit}
        )
        SELECT
            COUNT(*) as total_flights,
            COUNT(weather_obs_time_utc) as matched_weather,
            ROUND(AVG(weather_lag_minutes), 1) as avg_lag_min,
            ROUND(
                PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY weather_lag_minutes
                )::numeric, 1
            ) as median_lag_min,
            MAX(weather_lag_minutes) as max_lag_min,
            COUNT(weather_temp_f) as non_null_temp,
            COUNT(weather_visibility_sm) as non_null_vsby,
            COUNT(weather_wind_speed_kt) as non_null_wind,
            SUM(weather_has_precip) as precip_count,
            SUM(weather_has_thunderstorm) as ts_count,
            SUM(weather_has_snow) as snow_count
        FROM sample;
    """

    with database_connection() as conn:
        with conn.cursor() as cur:
            t0 = time.time()
            cur.execute(query)
            row = cur.fetchone()
            elapsed = time.time() - t0

    total = row[0]
    matched = row[1]
    metrics = {
        "sample_size": total,
        "matched_count": matched,
        "match_rate_pct": round(matched / total * 100, 2) if total else 0.0,
        "avg_lag_min": float(row[2]) if row[2] is not None else 0.0,
        "median_lag_min": float(row[3]) if row[3] is not None else 0.0,
        "max_lag_min": float(row[4]) if row[4] is not None else 0.0,
        "temp_non_null_pct": round(row[5] / total * 100, 2) if total else 0.0,
        "vsby_non_null_pct": round(row[6] / total * 100, 2) if total else 0.0,
        "wind_non_null_pct": round(row[7] / total * 100, 2) if total else 0.0,
        "precip_count": int(row[8]) if row[8] is not None else 0,
        "ts_count": int(row[9]) if row[9] is not None else 0,
        "snow_count": int(row[10]) if row[10] is not None else 0,
        "query_time_sec": round(elapsed, 2),
    }

    return metrics


def main() -> None:
    migration_path = Path("sql/007_create_top50_datasets.sql")
    top50_csv_path = Path("data/processed/weather/top_50_airports.csv")

    if not migration_path.exists():
        LOGGER.error("Migration 007 not found at: %s", migration_path)
        sys.exit(1)
    if not top50_csv_path.exists():
        LOGGER.error("Top 50 CSV not found at: %s", top50_csv_path)
        sys.exit(1)

    print("==================================================================")
    print("FlightOps ML — Step 3: Build & Verify Top 50 Modeling Datasets")
    print("==================================================================")

    # 1. Apply Migration 007
    apply_migration(migration_path)

    # 2. Populate Reference Table
    airport_count = populate_top_50_airports(top50_csv_path)

    # 3. Verify Splits
    split_results = verify_datasets()

    print("\n------------------------------------------------------------------")
    print(f"Top 50 Hubs Baseline Dataset Verification ({airport_count} Airports)")
    print("------------------------------------------------------------------")
    total_all = 0
    total_delayed = 0
    for split, data in split_results.items():
        total_all += int(data["baseline_total"])
        total_delayed += int(data["baseline_delayed"])
        print(
            f"  {split.capitalize():<12}: {data['baseline_total']:,} flights | "
            f"Delayed: {data['baseline_delayed']:,} ({data['baseline_rate_pct']}%)"
        )
    overall_rate = round(total_delayed / total_all * 100, 2)
    print(
        f"  Total 2025  : {total_all:,} flights | "
        f"Delayed: {total_delayed:,} ({overall_rate}%)"
    )

    # 4. Verify Weather Join
    metrics = verify_weather_join_metrics(sample_limit=100_000)

    n_sample = int(metrics["sample_size"])
    print("\n------------------------------------------------------------------")
    print(f"Weather Point-in-Time Join Audit ({n_sample:,} Flight Sample)")
    print("------------------------------------------------------------------")
    print(f"  Matched Observations:   {metrics['match_rate_pct']}%")
    print(
        f"  Freshness (Lag):        Average {metrics['avg_lag_min']} min, "
        f"Median {metrics['median_lag_min']} min (Max: {metrics['max_lag_min']} min)"
    )
    print(f"  Temperature Available:  {metrics['temp_non_null_pct']}%")
    print(f"  Visibility Available:   {metrics['vsby_non_null_pct']}%")
    print(f"  Wind Speed Available:   {metrics['wind_non_null_pct']}%")
    print(f"  Sample Precipitation:   {metrics['precip_count']:,} flights")
    print(f"  Sample Thunderstorms:   {metrics['ts_count']:,} flights")
    print(f"  Sample Snow:            {metrics['snow_count']:,} flights")
    print(f"  Lateral Join Throughput: {metrics['query_time_sec']}s for 100k rows")
    print("==================================================================")
    print("Step 3 Complete: Datasets are ready for Step 4 model training!")
    print("==================================================================")


if __name__ == "__main__":
    main()

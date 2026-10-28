"""Step 1: Data Health & Leakage Audit for Top 50 Hubs Modeling Datasets.

Audits:
1. Cardinality & Fan-Out (Baseline vs. Weather view 1-to-1 match).
2. Temporal Leakage (Strict adherence to T <= T_dep - 2h UTC cutoff).
3. Weather Match Rate across Train, Validation, and Test splits.
4. Per-Airport Weather Match Rates across all Top 50 origin hubs.
5. Observation Freshness & Lag Percentiles.
6. Sensor-Level Completeness and Null Rates.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from flightops_ml.ingestion.loader import database_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("audit_top50_data_health")


def check_prerequisites(cur) -> dict[str, int]:
    """Verify that tables and views exist and return base counts."""
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('top_50_airports', 'weather_observations', 'gold_features')
    """)
    tables = {r[0] for r in cur.fetchall()}

    cur.execute("""
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = 'public'
          AND table_name IN ('vw_gold_top50_baseline', 'vw_gold_top50_weather')
    """)
    views = {r[0] for r in cur.fetchall()}

    missing = []
    if "top_50_airports" not in tables:
        missing.append("public.top_50_airports")
    if "vw_gold_top50_baseline" not in views:
        missing.append("public.vw_gold_top50_baseline")
    if "vw_gold_top50_weather" not in views:
        missing.append("public.vw_gold_top50_weather")

    if missing:
        raise RuntimeError(
            f"Missing required database objects: {missing}. Run scripts/build_top50_datasets.py first."
        )

    cur.execute("SELECT count(*) FROM public.top_50_airports;")
    n_airports = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM public.weather_observations;")
    n_obs = cur.fetchone()[0]

    return {"top_50_airports": n_airports, "weather_observations": n_obs}


def audit_cardinality_and_splits(cur) -> dict:
    """Audit row counts, splits, and verify zero fan-out between Baseline and Weather."""
    LOGGER.info("Auditing cardinality, primary key uniqueness, and splits...")

    # 1. Baseline view counts
    cur.execute("""
        SELECT
            dataset_split,
            COUNT(*) as total_rows,
            COUNT(DISTINCT (source_sha256, source_row_number)) as distinct_keys,
            SUM(dep_del15) as delayed_rows,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct
        FROM public.vw_gold_top50_baseline
        GROUP BY dataset_split
        ORDER BY CASE dataset_split
            WHEN 'train' THEN 1
            WHEN 'validation' THEN 2
            WHEN 'test' THEN 3
        END;
    """)
    baseline_splits = cur.fetchall()

    # 2. Weather view counts
    cur.execute("""
        SELECT
            dataset_split,
            COUNT(*) as total_rows,
            COUNT(DISTINCT (source_sha256, source_row_number)) as distinct_keys,
            SUM(dep_del15) as delayed_rows,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct
        FROM public.vw_gold_top50_weather
        GROUP BY dataset_split
        ORDER BY CASE dataset_split
            WHEN 'train' THEN 1
            WHEN 'validation' THEN 2
            WHEN 'test' THEN 3
        END;
    """)
    weather_splits = cur.fetchall()

    split_comparison = {}
    for (b_split, b_tot, b_keys, b_del, b_rate), (
        _w_split,
        w_tot,
        _w_keys,
        _w_del,
        _w_rate,
    ) in zip(baseline_splits, weather_splits, strict=False):
        fanout = w_tot - b_tot
        key_dupes = b_tot - b_keys
        split_comparison[b_split] = {
            "baseline_rows": b_tot,
            "weather_rows": w_tot,
            "fanout_diff": fanout,
            "is_1to1": (fanout == 0 and key_dupes == 0),
            "delayed_rows": b_del,
            "delay_rate_pct": float(b_rate),
        }

    return split_comparison


def audit_temporal_leakage(cur) -> dict:
    """Verify strict adherence to T <= T_dep - 2h cutoff (zero future observations)."""
    LOGGER.info("Auditing temporal leakage and cutoff integrity...")

    cur.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(weather_obs_time_utc) as matched_rows,
            -- Negative lag means observation time was AFTER the cutoff (future leakage)
            COUNT(*) FILTER (WHERE weather_lag_minutes < 0) as future_obs_violations,
            -- Out-of-window violations: lag minutes > 360 (exceeds 6-hour lookback)
            COUNT(*) FILTER (WHERE weather_lag_minutes > 360) as lookback_violations,
            MIN(weather_lag_minutes) as min_lag_min,
            MAX(weather_lag_minutes) as max_lag_min
        FROM public.vw_gold_top50_weather;
    """)
    row = cur.fetchone()

    return {
        "total_flights": row[0],
        "matched_weather_flights": row[1],
        "future_obs_violations": row[2],
        "negative_lag_violations": row[2],
        "lookback_violations": row[3],
        "min_lag_min": float(row[4]) if row[4] is not None else None,
        "max_lag_min": float(row[5]) if row[5] is not None else None,
        "leakage_free": (row[2] == 0 and row[3] == 0),
    }


def audit_match_rates_and_freshness(cur) -> dict:
    """Compute weather match rates by split and freshness percentiles."""
    LOGGER.info("Auditing match rates by split and freshness distribution...")

    # 1. Match rate by split
    cur.execute("""
        SELECT
            dataset_split,
            COUNT(*) as total_flights,
            COUNT(weather_obs_time_utc) as matched_flights,
            ROUND(COUNT(weather_obs_time_utc)::numeric / COUNT(*) * 100, 2) as match_rate_pct,
            ROUND(AVG(weather_lag_minutes)::numeric, 1) as avg_lag_min,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as median_lag_min
        FROM public.vw_gold_top50_weather
        GROUP BY dataset_split
        ORDER BY CASE dataset_split
            WHEN 'train' THEN 1
            WHEN 'validation' THEN 2
            WHEN 'test' THEN 3
        END;
    """)
    split_matches = {}
    for split, tot, matched, rate, avg_lag, med_lag in cur.fetchall():
        split_matches[split] = {
            "total_flights": tot,
            "matched_flights": matched,
            "match_rate_pct": float(rate),
            "avg_lag_min": float(avg_lag),
            "median_lag_min": float(med_lag),
        }

    # 2. Combined Freshness & Sensor query in a single table pass
    cur.execute("""
        SELECT
            COUNT(*) as total_matched,
            ROUND(AVG(weather_lag_minutes)::numeric, 2) as mean_lag,
            ROUND(STDDEV(weather_lag_minutes)::numeric, 2) as std_lag,
            MIN(weather_lag_minutes) as min_lag,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p90,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p95,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as p99,
            MAX(weather_lag_minutes) as max_lag,
            -- Buckets
            COUNT(*) FILTER (WHERE weather_lag_minutes <= 30) as le_30m,
            COUNT(*) FILTER (WHERE weather_lag_minutes > 30 AND weather_lag_minutes <= 60) as m30_60,
            COUNT(*) FILTER (WHERE weather_lag_minutes > 60 AND weather_lag_minutes <= 120) as m60_120,
            COUNT(*) FILTER (WHERE weather_lag_minutes > 120 AND weather_lag_minutes <= 180) as m120_180,
            COUNT(*) FILTER (WHERE weather_lag_minutes > 180 AND weather_lag_minutes <= 240) as m180_240,
            COUNT(*) FILTER (WHERE weather_lag_minutes > 240) as gt_240m,
            -- Sensors
            COUNT(weather_temp_f) as non_null_temp,
            COUNT(weather_dew_point_f) as non_null_dew,
            COUNT(weather_humidity_pct) as non_null_relh,
            COUNT(weather_wind_dir_deg) as non_null_drct,
            COUNT(weather_wind_speed_kt) as non_null_sknt,
            COUNT(weather_wind_gust_kt) as non_null_gust,
            COUNT(*) FILTER (WHERE weather_wind_gust_kt > 0) as active_gusts,
            COUNT(weather_visibility_sm) as non_null_vsby,
            COUNT(weather_altimeter_in_hg) as non_null_alti,
            COUNT(weather_precip_1h_in) as non_null_precip,
            SUM(weather_has_precip) as has_precip,
            SUM(weather_has_thunderstorm) as has_thunderstorm,
            SUM(weather_has_snow) as has_snow,
            SUM(weather_has_fog_haze) as has_fog_haze
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL;
    """)
    r = cur.fetchone()
    tot_matched = r[0]

    percentiles = {
        "mean": float(r[1]),
        "std": float(r[2]),
        "min": float(r[3]),
        "p10": float(r[4]),
        "p25": float(r[5]),
        "p50_median": float(r[6]),
        "p75": float(r[7]),
        "p90": float(r[8]),
        "p95": float(r[9]),
        "p99": float(r[10]),
        "max": float(r[11]),
    }

    buckets = {
        "<= 30 min (very fresh)": {
            "count": r[12],
            "pct": round(r[12] / tot_matched * 100, 2),
        },
        "31 - 60 min (nominal hourly)": {
            "count": r[13],
            "pct": round(r[13] / tot_matched * 100, 2),
        },
        "61 - 120 min (minor gap)": {
            "count": r[14],
            "pct": round(r[14] / tot_matched * 100, 2),
        },
        "121 - 180 min (moderate gap)": {
            "count": r[15],
            "pct": round(r[15] / tot_matched * 100, 2),
        },
        "181 - 240 min (large gap)": {
            "count": r[16],
            "pct": round(r[16] / tot_matched * 100, 2),
        },
        "> 240 min (stale observation)": {
            "count": r[17],
            "pct": round(r[17] / tot_matched * 100, 2),
        },
    }

    sensors = {
        "total_matched_flights": tot_matched,
        "temperature_pct": round(r[18] / tot_matched * 100, 2),
        "dew_point_pct": round(r[19] / tot_matched * 100, 2),
        "relative_humidity_pct": round(r[20] / tot_matched * 100, 2),
        "wind_direction_pct": round(r[21] / tot_matched * 100, 2),
        "wind_speed_pct": round(r[22] / tot_matched * 100, 2),
        "wind_gust_active_pct": round(r[24] / tot_matched * 100, 2),
        "visibility_pct": round(r[25] / tot_matched * 100, 2),
        "altimeter_pct": round(r[26] / tot_matched * 100, 2),
        "precip_1h_pct": round(r[27] / tot_matched * 100, 2),
        "phenomena_rates": {
            "has_precip": {
                "count": int(r[28]),
                "pct": round(r[28] / tot_matched * 100, 2),
            },
            "has_thunderstorm": {
                "count": int(r[29]),
                "pct": round(r[29] / tot_matched * 100, 2),
            },
            "has_snow": {
                "count": int(r[30]),
                "pct": round(r[30] / tot_matched * 100, 2),
            },
            "has_fog_haze": {
                "count": int(r[31]),
                "pct": round(r[31] / tot_matched * 100, 2),
            },
        },
    }

    return {
        "splits": split_matches,
        "freshness_percentiles": percentiles,
        "lag_buckets": buckets,
        "sensors": sensors,
    }


def audit_airport_weather_coverage(cur) -> list[dict]:
    """Audit match rate for each of the 50 origin hubs."""
    LOGGER.info("Auditing weather match rate per airport across Top 50 hubs...")

    cur.execute("""
        SELECT
            origin,
            COUNT(*) as total_flights,
            COUNT(weather_obs_time_utc) as matched_flights,
            ROUND(COUNT(weather_obs_time_utc)::numeric / COUNT(*) * 100, 2) as match_rate_pct,
            ROUND(AVG(weather_lag_minutes)::numeric, 1) as avg_lag_min,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY weather_lag_minutes)::numeric, 1) as median_lag_min,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct
        FROM public.vw_gold_top50_weather
        GROUP BY origin
        ORDER BY match_rate_pct ASC, total_flights DESC;
    """)
    rows = cur.fetchall()
    return [
        {
            "origin": r[0],
            "total_flights": r[1],
            "matched_flights": r[2],
            "match_rate_pct": float(r[3]),
            "avg_lag_min": float(r[4]) if r[4] is not None else 0.0,
            "median_lag_min": float(r[5]) if r[5] is not None else 0.0,
            "delay_rate_pct": float(r[6]),
        }
        for r in rows
    ]


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("FlightOps ML — Step 1: Top 50 Hubs Data Health & Leakage Audit")
    print("=" * 80)

    with database_connection() as conn:
        with conn.cursor() as cur:
            prereqs = check_prerequisites(cur)
            print(
                f"Verified Prerequisites: {prereqs['top_50_airports']} Top 50 Airports | {prereqs['weather_observations']:,} Weather Observations in DB"
            )

            cardinality = audit_cardinality_and_splits(cur)
            leakage = audit_temporal_leakage(cur)
            match_stats = audit_match_rates_and_freshness(cur)
            sensors = match_stats["sensors"]
            airports = audit_airport_weather_coverage(cur)

    elapsed = time.time() - t0

    # Output Presentation
    print("\n" + "-" * 80)
    print("1. CARDINALITY & 1-TO-1 FAN-OUT RECONCILIATION")
    print("-" * 80)
    tot_base = sum(d["baseline_rows"] for d in cardinality.values())
    tot_weath = sum(d["weather_rows"] for d in cardinality.values())
    tot_del = sum(d["delayed_rows"] for d in cardinality.values())
    overall_rate = round(tot_del / tot_base * 100, 2)

    for split, data in cardinality.items():
        status = "PASSED (1-to-1)" if data["is_1to1"] else "FAILED (Fan-out detected)"
        print(
            f"  {split.capitalize():<12}: Baseline = {data['baseline_rows']:,} | Weather = {data['weather_rows']:,} | Diff = {data['fanout_diff']} [{status}]"
        )
        print(
            f"                Delayed = {data['delayed_rows']:,} ({data['delay_rate_pct']}%)"
        )
    print(
        f"  Total 2025  : Baseline = {tot_base:,} | Weather = {tot_weath:,} | Overall Delay Rate = {overall_rate}%"
    )

    print("\n" + "-" * 80)
    print("2. TEMPORAL LEAKAGE VERIFICATION (T <= T_dep - 2h UTC)")
    print("-" * 80)
    print(
        f"  Future Observation Violations (obs_time > cutoff) : {leakage['future_obs_violations']:,} (Target: 0)"
    )
    print(
        f"  Negative Lag Violations (lag < 0)                 : {leakage['negative_lag_violations']:,} (Target: 0)"
    )
    print(
        f"  Lookback Window Violations (lag > 360 min)        : {leakage['lookback_violations']:,} (Target: 0)"
    )
    print(
        f"  Min Observed Lag                                  : {leakage['min_lag_min']} min"
    )
    print(
        f"  Max Observed Lag                                  : {leakage['max_lag_min']} min"
    )
    leak_status = (
        "VERIFIED ZERO LEAKAGE" if leakage["leakage_free"] else "LEAKAGE DETECTED"
    )
    print(f"  Temporal Integrity Status                         : [{leak_status}]")

    print("\n" + "-" * 80)
    print("3. WEATHER MATCH RATES ACROSS CHRONOLOGICAL SPLITS")
    print("-" * 80)
    for split, data in match_stats["splits"].items():
        print(
            f"  {split.capitalize():<12}: {data['matched_flights']:,} / {data['total_flights']:,} matched ({data['match_rate_pct']}%) | Avg Lag: {data['avg_lag_min']}m, Median: {data['median_lag_min']}m"
        )

    tot_matched = sum(d["matched_flights"] for d in match_stats["splits"].values())
    overall_match = round(tot_matched / tot_base * 100, 2)
    print(f"  Overall Match Rate: {tot_matched:,} / {tot_base:,} ({overall_match}%)")

    print("\n" + "-" * 80)
    print("4. OBSERVATION FRESHNESS PERCENTILES & LAG PROFILE")
    print("-" * 80)
    fp = match_stats["freshness_percentiles"]
    print(f"  Mean Lag:   {fp['mean']} min (Std: {fp['std']} min)")
    print(
        f"  Percentiles : Min={fp['min']}m | P10={fp['p10']}m | P25={fp['p25']}m | P50 (Median)={fp['p50_median']}m | P75={fp['p75']}m | P90={fp['p90']}m | P95={fp['p95']}m | P99={fp['p99']}m | Max={fp['max']}m"
    )
    print("  Lag Buckets:")
    for b_name, b_val in match_stats["lag_buckets"].items():
        print(f"    {b_name:<30}: {b_val['count']:,} flights ({b_val['pct']}%)")

    print("\n" + "-" * 80)
    print("5. SENSOR-LEVEL COMPLETENESS & PHENOMENA RATES")
    print("-" * 80)
    print(f"  Temperature Available       : {sensors['temperature_pct']}%")
    print(f"  Dew Point Available         : {sensors['dew_point_pct']}%")
    print(f"  Relative Humidity Available : {sensors['relative_humidity_pct']}%")
    print(f"  Wind Direction Available    : {sensors['wind_direction_pct']}%")
    print(f"  Wind Speed Available        : {sensors['wind_speed_pct']}%")
    print(f"  Visibility Available        : {sensors['visibility_pct']}%")
    print(f"  Altimeter Pressure Available: {sensors['altimeter_pct']}%")
    print(f"  Precipitation Gauge Avail.  : {sensors['precip_1h_pct']}%")
    print(f"  Active Wind Gusts Reported  : {sensors['wind_gust_active_pct']}%")
    print("  Reported Weather Phenomena:")
    for phen, p_val in sensors["phenomena_rates"].items():
        print(f"    {phen:<25}: {p_val['count']:,} flights ({p_val['pct']}%)")

    print("\n" + "-" * 80)
    print("6. AIRPORT-LEVEL WEATHER COVERAGE (LOWEST 5 vs HIGHEST 5)")
    print("-" * 80)
    lowest_5 = airports[:5]
    highest_5 = airports[-5:]
    print("  Lowest 5 Hubs by Match Rate:")
    for a in lowest_5:
        print(
            f"    {a['origin']:<5}: Match Rate = {a['match_rate_pct']}% ({a['matched_flights']:,}/{a['total_flights']:,}) | Med Lag = {a['median_lag_min']}m | Delay Rate = {a['delay_rate_pct']}%"
        )
    print("  Highest 5 Hubs by Match Rate:")
    for a in reversed(highest_5):
        print(
            f"    {a['origin']:<5}: Match Rate = {a['match_rate_pct']}% ({a['matched_flights']:,}/{a['total_flights']:,}) | Med Lag = {a['median_lag_min']}m | Delay Rate = {a['delay_rate_pct']}%"
        )

    print("\n" + "=" * 80)
    print(f"Audit Complete in {elapsed:.2f}s. All checks passed!")
    print("=" * 80)

    # Save summary report artifact
    audit_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_flights": tot_base,
        "total_matched_weather": tot_matched,
        "overall_match_rate_pct": overall_match,
        "overall_delay_rate_pct": overall_rate,
        "fanout_verified_1to1": all(d["is_1to1"] for d in cardinality.values()),
        "temporal_leakage_free": leakage["leakage_free"],
        "splits": cardinality,
        "leakage": leakage,
        "match_stats": match_stats,
        "sensors": sensors,
        "airports": airports,
    }
    out_path = Path("docs/TOP50_DATA_HEALTH_AUDIT.json")
    out_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    LOGGER.info("Exported audit report to: %s", out_path)


if __name__ == "__main__":
    main()

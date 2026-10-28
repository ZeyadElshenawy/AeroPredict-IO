"""Extract Top 50 Hubs with display names and coordinates to static/hubs.json."""

from __future__ import annotations

import json
from pathlib import Path

from flightops_ml.ingestion.loader import database_connection


def main() -> None:
    with database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    b.origin as iata,
                    COALESCE(MAX(a.display_airport_name), b.origin) as name,
                    ROUND(AVG(b.origin_latitude)::numeric, 4) as lat,
                    ROUND(AVG(b.origin_longitude)::numeric, 4) as lon,
                    COUNT(*) as flight_count,
                    ROUND(AVG(b.dep_del15)::numeric * 100, 2) as delay_rate_pct
                FROM public.vw_gold_top50_baseline b
                LEFT JOIN public.airport_reference a
                    ON b.origin = a.airport_code AND a.is_latest = TRUE
                GROUP BY b.origin
                ORDER BY flight_count DESC;
            """)
            rows = cur.fetchall()

    hubs = []
    for r in rows:
        hubs.append(
            {
                "iata": r[0],
                "name": r[1],
                "lat": float(r[2]) if r[2] is not None else 0.0,
                "lon": float(r[3]) if r[3] is not None else 0.0,
                "flight_count": int(r[4]),
                "delay_rate_pct": float(r[5]),
            }
        )

    out_path = Path("src/flightops_ml/service/static/hubs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hubs, f, indent=2)

    print(f"Extracted {len(hubs)} hubs to {out_path}")
    for h in hubs[:5]:
        print(
            f"  - {h['iata']}: {h['name']} ({h['flight_count']} flights, {h['delay_rate_pct']}%)"
        )


if __name__ == "__main__":
    main()

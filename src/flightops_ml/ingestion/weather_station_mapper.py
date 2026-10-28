"""Weather station mapping and inventory orchestration for FlightOps ML.

This module maps origin airports from gold_features to candidate weather
stations in the Iowa Environmental Mesonet (IEM) ASOS/AWOS station catalogue.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flightops_ml.ingestion.loader import database_connection

EARTH_RADIUS_KM = 6371.0088

EXPECTED_INVENTORY_COLUMNS = {
    "origin_airport_id",
    "latitude",
    "longitude",
    "flight_count",
    "active_days",
}

EXPECTED_CATALOGUE_COLUMNS = {
    "stid",
    "station_name",
    "lat",
    "lon",
    "elev",
    "begints",
    "endts",
    "iem_network",
}


@dataclass(frozen=True)
class AirportRecord:
    origin_airport_id: int
    airport_code: str
    airport_name: str
    latitude: float
    longitude: float
    flight_count: int
    active_days: int


@dataclass(frozen=True)
class WeatherStationRecord:
    stid: str
    station_name: str
    latitude: float
    longitude: float
    elevation_m: float | None
    begints: str
    endts: str
    iem_network: str


@dataclass(frozen=True)
class CandidateMatch:
    airport_id: int
    airport_code: str
    candidate_rank: int
    station_id: str
    station_name: str
    station_latitude: float
    station_longitude: float
    elevation_m: float | None
    distance_km: float
    network: str
    archive_begin: str
    archive_end: str
    is_exact_code: bool
    is_icao_match: bool
    archive_covers_2025: bool
    is_selected: bool


@dataclass(frozen=True)
class AirportMapping:
    origin_airport_id: int
    airport_code: str
    airport_name: str
    airport_latitude: float
    airport_longitude: float
    station_id: str
    station_name: str
    station_latitude: float
    station_longitude: float
    elevation_m: float | None
    distance_km: float
    match_type: str
    selection_reason: str
    network: str
    archive_begin: str
    archive_end: str


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in decimal degrees."""
    if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0):
        raise ValueError(f"Latitude out of range [-90, 90]: lat1={lat1}, lat2={lat2}")
    if not (-180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0):
        raise ValueError(
            f"Longitude out of range [-180, 180]: lon1={lon1}, lon2={lon2}"
        )

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    # Clamp to [0, 1] to avoid float precision domain error in asin
    a_clamped = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a_clamped))


def compute_sha256(path: Path) -> str:
    """Compute the SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_airport_inventory(
    output_path: Path,
    con=None,
    expected_count: int = 352,
    expected_flights: int = 6879484,
) -> dict[str, Any]:
    """Query PostgreSQL and export origin airport inventory to CSV.

    Validates that exactly 352 unique origin airport IDs are present and
    the total flight count equals 6,879,484.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = """
    SELECT
        origin_airport_id,
        MIN(origin_latitude) AS latitude,
        MIN(origin_longitude) AS longitude,
        COUNT(*) AS flight_count,
        COUNT(DISTINCT flight_date) AS active_days
    FROM public.gold_features
    GROUP BY origin_airport_id
    ORDER BY origin_airport_id;
    """

    def _execute(conn) -> list[tuple]:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    if con is None:
        with database_connection() as managed_conn:
            rows = _execute(managed_conn)
    else:
        rows = _execute(con)

    actual_count = len(rows)
    total_flights = sum(int(r[3]) for r in rows)

    if actual_count != expected_count:
        raise ValueError(
            f"Airport inventory count mismatch: expected {expected_count}, "
            f"got {actual_count}"
        )
    if total_flights != expected_flights:
        raise ValueError(
            f"Flight count mismatch: expected {expected_flights}, got {total_flights}"
        )

    # Validate coordinate completeness
    for row in rows:
        aid, lat, lon, _fcnt, _adays = row
        if lat is None or lon is None:
            raise ValueError(
                f"Origin airport {aid} has null coordinates: lat={lat}, lon={lon}"
            )
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            raise ValueError(
                f"Origin airport {aid} has invalid coordinates: ({lat}, {lon})"
            )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "origin_airport_id",
                "latitude",
                "longitude",
                "flight_count",
                "active_days",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row[0],
                    f"{float(row[1]):.6f}",
                    f"{float(row[2]):.6f}",
                    row[3],
                    row[4],
                ]
            )

    return {
        "output_path": str(output_path),
        "record_count": actual_count,
        "total_flights": total_flights,
        "sha256": compute_sha256(output_path),
    }


def load_airport_inventory(path: Path) -> list[dict[str, Any]]:
    """Load and validate the airport inventory CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Airport inventory file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = EXPECTED_INVENTORY_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"Airport inventory missing required columns: {sorted(missing)}"
            )

        records = []
        seen_ids = set()
        for i, row in enumerate(reader, start=2):
            raw_id = row.get("origin_airport_id", "").strip()
            if not raw_id:
                raise ValueError(f"Row {i}: Missing origin_airport_id")
            try:
                aid = int(raw_id)
            except ValueError as err:
                raise ValueError(
                    f"Row {i}: Invalid integer origin_airport_id '{raw_id}'"
                ) from err

            if aid in seen_ids:
                raise ValueError(f"Row {i}: Duplicate origin_airport_id {aid}")
            seen_ids.add(aid)

            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (ValueError, KeyError, TypeError) as err:
                raise ValueError(
                    f"Row {i} (airport {aid}): Invalid coordinates "
                    f"({row.get('latitude')}, {row.get('longitude')})"
                ) from err

            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Row {i} (airport {aid}): Coordinates out of range ({lat}, {lon})"
                )

            try:
                fcnt = int(row["flight_count"])
                adays = int(row["active_days"])
            except (ValueError, KeyError, TypeError) as err:
                raise ValueError(f"Row {i} (airport {aid}): Invalid counts") from err

            if fcnt <= 0 or adays <= 0:
                raise ValueError(
                    f"Row {i} (airport {aid}): Nonpositive flight count or active days"
                )

            records.append(
                {
                    "origin_airport_id": aid,
                    "latitude": lat,
                    "longitude": lon,
                    "flight_count": fcnt,
                    "active_days": adays,
                    "airport_code": row.get("airport_code", "").strip().upper(),
                    "airport_name": row.get("airport_name", "").strip(),
                }
            )

    return records


def load_station_catalogue(path: Path) -> list[WeatherStationRecord]:
    """Load and validate the IEM station catalogue CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather station catalogue not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = EXPECTED_CATALOGUE_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Catalogue missing required columns: {sorted(missing)}")

        stations: list[WeatherStationRecord] = []
        seen_stids: dict[str, int] = {}

        for i, row in enumerate(reader, start=2):
            raw_stid = row.get("stid", "")
            # Preserve leading zeros, strip whitespace, uppercase
            stid = str(raw_stid).strip().upper()
            if not stid:
                raise ValueError(f"Row {i}: Empty station identifier (stid)")

            if stid in seen_stids:
                prev_line = seen_stids[stid]
                raise ValueError(
                    f"Row {i}: Duplicate station identifier '{stid}' "
                    f"(previously seen at row {prev_line})"
                )
            seen_stids[stid] = i

            name = row.get("station_name", "").strip()
            raw_lat = row.get("lat", "").strip()
            raw_lon = row.get("lon", "").strip()
            if not raw_lat or not raw_lon:
                raise ValueError(f"Row {i} (station {stid}): Missing coordinates")

            try:
                lat = float(raw_lat)
                lon = float(raw_lon)
            except ValueError as err:
                raise ValueError(
                    f"Row {i} (station {stid}): Invalid float coordinates "
                    f"({raw_lat}, {raw_lon})"
                ) from err

            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Row {i} (station {stid}): Coordinates out of range ({lat}, {lon})"
                )

            elev_str = row.get("elev", "").strip()
            elevation: float | None = None
            if elev_str:
                try:
                    elevation = float(elev_str)
                except ValueError:
                    elevation = None

            stations.append(
                WeatherStationRecord(
                    stid=stid,
                    station_name=name,
                    latitude=lat,
                    longitude=lon,
                    elevation_m=elevation,
                    begints=row.get("begints", "").strip(),
                    endts=row.get("endts", "").strip(),
                    iem_network=row.get("iem_network", "").strip().upper(),
                )
            )

    return stations


def resolve_airport_metadata(
    airport_ids: list[int],
    con=None,
    master_cord_path: Path | None = None,
) -> dict[int, tuple[str, str]]:
    """Resolve airport code and name for origin_airport_ids.

    Tries PostgreSQL first, then falls back to local T_MASTER_CORD.csv.
    """
    metadata: dict[int, tuple[str, str]] = {}
    remaining_ids = set(airport_ids)

    # 1. Try PostgreSQL if available
    try:

        def _fetch_pg(conn):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH gold_airports AS (
                        SELECT UNNEST(%s::bigint[]) AS origin_airport_id
                    ),
                    silver_origins AS (
                        SELECT DISTINCT origin_airport_id, origin
                        FROM public.silver_flights
                    ),
                    point_in_time_ar AS (
                        SELECT DISTINCT ON (airport_id)
                            airport_id,
                            airport_code,
                            display_airport_name
                        FROM public.airport_reference
                        WHERE airport_start_date <= '2025-12-31'
                          AND (airport_thru_date IS NULL
                               OR airport_thru_date >= '2025-01-01')
                        ORDER BY airport_id, is_latest DESC
                    )
                    SELECT
                        ga.origin_airport_id,
                        COALESCE(so.origin, ar.airport_code, '') AS code,
                        COALESCE(ar.display_airport_name, '') AS name
                    FROM gold_airports ga
                    LEFT JOIN silver_origins so
                      ON ga.origin_airport_id = so.origin_airport_id
                    LEFT JOIN point_in_time_ar ar
                      ON ga.origin_airport_id = ar.airport_id;
                    """,
                    (list(airport_ids),),
                )
                return cur.fetchall()

        if con is None:
            with database_connection() as managed_conn:
                rows = _fetch_pg(managed_conn)
        else:
            rows = _fetch_pg(con)

        for aid, code, name in rows:
            if code or name:
                metadata[int(aid)] = (code.strip().upper(), name.strip())
                remaining_ids.discard(int(aid))

    except Exception:
        # DB not available or offline; proceed to fallback
        pass

    # 2. Try local T_MASTER_CORD.csv for any remaining IDs
    if remaining_ids:
        default_cord_paths = [
            Path("data/raw/airport_reference/T_MASTER_CORD.csv"),
            Path("data/airport_reference/T_MASTER_CORD.csv"),
        ]
        if master_cord_path is not None:
            default_cord_paths.insert(0, Path(master_cord_path))

        cord_file = next((p for p in default_cord_paths if p.exists()), None)
        if cord_file is not None:
            with cord_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        aid = int(row.get("AIRPORT_ID", 0))
                    except (ValueError, TypeError):
                        continue
                    if aid in remaining_ids:
                        code = row.get("AIRPORT", "").strip().upper()
                        name = row.get("DISPLAY_AIRPORT_NAME", "").strip()
                        start_date = row.get("AIRPORT_START_DATE", "")
                        thru_date = row.get("AIRPORT_THRU_DATE", "")
                        is_2025 = (
                            "2025" in thru_date
                            or "2026" in thru_date
                            or thru_date == ""
                        ) and ("2026" not in start_date or "DJT" not in code)
                        if aid not in metadata or is_2025:
                            metadata[aid] = (code, name)

    return metadata


def is_active_during_2025(begints: str, endts: str) -> bool:
    """Check whether station archive covers the 2025 calendar year."""
    has_begin = begints != "" and begints <= "2025-01-01 23:59"
    has_end = endts == "" or endts >= "2025-12-31 00:00"
    return has_begin and has_end


def find_candidate_stations(
    airport: AirportRecord,
    stations: list[WeatherStationRecord],
    max_candidates: int = 5,
    search_radius_km: float = 100.0,
) -> list[CandidateMatch]:
    """Find and rank candidate weather stations for an airport by distance."""
    scored: list[tuple[float, WeatherStationRecord]] = []
    airport_code = airport.airport_code.upper()

    # Known territorial prefixes
    standard_icao_prefixes = ("K", "P", "PA", "PH", "TJ", "TI", "PG", "NS")
    expected_icao_codes = {pfx + airport_code for pfx in standard_icao_prefixes}
    if airport_code == "PPG":
        expected_icao_codes.add("NSTU")
    if airport_code == "SPN":
        expected_icao_codes.add("PGSN")

    for st in stations:
        dist = haversine_distance_km(
            airport.latitude, airport.longitude, st.latitude, st.longitude
        )
        if dist <= search_radius_km:
            scored.append((dist, st))

    if not scored:
        nearest_st = min(
            stations,
            key=lambda s: haversine_distance_km(
                airport.latitude, airport.longitude, s.latitude, s.longitude
            ),
        )
        dist = haversine_distance_km(
            airport.latitude,
            airport.longitude,
            nearest_st.latitude,
            nearest_st.longitude,
        )
        scored.append((dist, nearest_st))

    scored.sort(key=lambda x: x[0])
    top_candidates = scored[:max_candidates]

    matches: list[CandidateMatch] = []
    for rank, (dist, st) in enumerate(top_candidates, start=1):
        is_exact = bool(airport_code and st.stid == airport_code)
        is_icao = bool(airport_code and st.stid in expected_icao_codes and not is_exact)
        covers_2025 = is_active_during_2025(st.begints, st.endts)

        matches.append(
            CandidateMatch(
                airport_id=airport.origin_airport_id,
                airport_code=airport_code,
                candidate_rank=rank,
                station_id=st.stid,
                station_name=st.station_name,
                station_latitude=st.latitude,
                station_longitude=st.longitude,
                elevation_m=st.elevation_m,
                distance_km=round(dist, 3),
                network=st.iem_network,
                archive_begin=st.begints,
                archive_end=st.endts,
                is_exact_code=is_exact,
                is_icao_match=is_icao,
                archive_covers_2025=covers_2025,
                is_selected=False,
            )
        )

    return matches


def select_best_station(
    airport: AirportRecord,
    candidates: list[CandidateMatch],
    on_airport_threshold_km: float = 5.0,
    max_allowable_distance_km: float = 30.0,
) -> AirportMapping:
    """Select the best candidate weather station based on evidence and provenance."""
    if not candidates:
        raise ValueError(
            f"No candidate stations available for airport {airport.origin_airport_id}"
        )

    on_airport = [
        c
        for c in candidates
        if c.distance_km <= on_airport_threshold_km and c.archive_covers_2025
    ]

    # 1. Exact IATA match physically on airport
    exact_on_airport = next((c for c in on_airport if c.is_exact_code), None)
    if exact_on_airport:
        reason = (
            f"Exact IATA station identifier match on airport field "
            f"({exact_on_airport.distance_km:.2f} km) with active 2025 archive"
        )
        return AirportMapping(
            origin_airport_id=airport.origin_airport_id,
            airport_code=airport.airport_code,
            airport_name=airport.airport_name,
            airport_latitude=airport.latitude,
            airport_longitude=airport.longitude,
            station_id=exact_on_airport.station_id,
            station_name=exact_on_airport.station_name,
            station_latitude=exact_on_airport.station_latitude,
            station_longitude=exact_on_airport.station_longitude,
            elevation_m=exact_on_airport.elevation_m,
            distance_km=exact_on_airport.distance_km,
            match_type="EXACT_IATA_ON_AIRPORT",
            selection_reason=reason,
            network=exact_on_airport.network,
            archive_begin=exact_on_airport.archive_begin,
            archive_end=exact_on_airport.archive_end,
        )

    # 2. Territorial ICAO prefix match physically on airport
    icao_on_airport = next((c for c in on_airport if c.is_icao_match), None)
    if icao_on_airport:
        reason = (
            f"Territorial ICAO station identifier match on airport field "
            f"({icao_on_airport.distance_km:.2f} km) with active 2025 archive"
        )
        return AirportMapping(
            origin_airport_id=airport.origin_airport_id,
            airport_code=airport.airport_code,
            airport_name=airport.airport_name,
            airport_latitude=airport.latitude,
            airport_longitude=airport.longitude,
            station_id=icao_on_airport.station_id,
            station_name=icao_on_airport.station_name,
            station_latitude=icao_on_airport.station_latitude,
            station_longitude=icao_on_airport.station_longitude,
            elevation_m=icao_on_airport.elevation_m,
            distance_km=icao_on_airport.distance_km,
            match_type="ICAO_PREFIX_ON_AIRPORT",
            selection_reason=reason,
            network=icao_on_airport.network,
            archive_begin=icao_on_airport.archive_begin,
            archive_end=icao_on_airport.archive_end,
        )

    # 3. Colocated airfield ASOS station (FAA LID or colocated on airfield)
    if on_airport:
        colocated = on_airport[0]
        reason = (
            f"Colocated airfield ASOS station verified by coordinates/facility "
            f"name ({colocated.distance_km:.2f} km) with active 2025 archive"
        )
        return AirportMapping(
            origin_airport_id=airport.origin_airport_id,
            airport_code=airport.airport_code,
            airport_name=airport.airport_name,
            airport_latitude=airport.latitude,
            airport_longitude=airport.longitude,
            station_id=colocated.station_id,
            station_name=colocated.station_name,
            station_latitude=colocated.station_latitude,
            station_longitude=colocated.station_longitude,
            elevation_m=colocated.elevation_m,
            distance_km=colocated.distance_km,
            match_type="AIRPORT_FACILITY_COORDINATE_MATCH",
            selection_reason=reason,
            network=colocated.network,
            archive_begin=colocated.archive_begin,
            archive_end=colocated.archive_end,
        )

    # 4. Nearby fallback station within allowable distance (5 to 30 km)
    vicinity = [
        c
        for c in candidates
        if c.distance_km <= max_allowable_distance_km and c.archive_covers_2025
    ]
    if vicinity:
        nearest_vicinity = vicinity[0]
        reason = (
            f"Nearby regional ASOS fallback station "
            f"({nearest_vicinity.distance_km:.2f} km) with active 2025 archive"
        )
        return AirportMapping(
            origin_airport_id=airport.origin_airport_id,
            airport_code=airport.airport_code,
            airport_name=airport.airport_name,
            airport_latitude=airport.latitude,
            airport_longitude=airport.longitude,
            station_id=nearest_vicinity.station_id,
            station_name=nearest_vicinity.station_name,
            station_latitude=nearest_vicinity.station_latitude,
            station_longitude=nearest_vicinity.station_longitude,
            elevation_m=nearest_vicinity.elevation_m,
            distance_km=nearest_vicinity.distance_km,
            match_type="NEARBY_FALLBACK_VICINITY",
            selection_reason=reason,
            network=nearest_vicinity.network,
            archive_begin=nearest_vicinity.archive_begin,
            archive_end=nearest_vicinity.archive_end,
        )

    # 5. Flagged for manual review (distant fallback or missing 2025 archive)
    top_cand = candidates[0]
    reason = (
        f"No defensible candidate within {max_allowable_distance_km} km "
        f"with active 2025 archive (nearest is {top_cand.distance_km:.2f} km)"
    )
    return AirportMapping(
        origin_airport_id=airport.origin_airport_id,
        airport_code=airport.airport_code,
        airport_name=airport.airport_name,
        airport_latitude=airport.latitude,
        airport_longitude=airport.longitude,
        station_id=top_cand.station_id,
        station_name=top_cand.station_name,
        station_latitude=top_cand.station_latitude,
        station_longitude=top_cand.station_longitude,
        elevation_m=top_cand.elevation_m,
        distance_km=top_cand.distance_km,
        match_type="FLAGGED_FOR_MANUAL_REVIEW",
        selection_reason=reason,
        network=top_cand.network,
        archive_begin=top_cand.archive_begin,
        archive_end=top_cand.archive_end,
    )


def map_airports_to_weather_stations(
    inventory_path: Path,
    catalogue_path: Path,
    output_dir: Path,
    con=None,
    master_cord_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the complete airport-to-weather-station mapping workflow."""
    inventory_path = Path(inventory_path)
    catalogue_path = Path(catalogue_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_inventory = load_airport_inventory(inventory_path)
    stations = load_station_catalogue(catalogue_path)

    airport_ids = [r["origin_airport_id"] for r in raw_inventory]
    metadata_map = resolve_airport_metadata(
        airport_ids, con=con, master_cord_path=master_cord_path
    )

    airports: list[AirportRecord] = []
    for r in raw_inventory:
        aid = r["origin_airport_id"]
        code, name = metadata_map.get(
            aid, (r.get("airport_code", ""), r.get("airport_name", ""))
        )
        airports.append(
            AirportRecord(
                origin_airport_id=aid,
                airport_code=code,
                airport_name=name,
                latitude=r["latitude"],
                longitude=r["longitude"],
                flight_count=r["flight_count"],
                active_days=r["active_days"],
            )
        )

    all_mappings: list[AirportMapping] = []
    all_candidates: list[CandidateMatch] = []

    for airport in airports:
        candidates = find_candidate_stations(airport, stations, max_candidates=5)
        best_mapping = select_best_station(airport, candidates)
        all_mappings.append(best_mapping)

        for c in candidates:
            is_sel = c.station_id == best_mapping.station_id
            all_candidates.append(
                CandidateMatch(
                    airport_id=c.airport_id,
                    airport_code=c.airport_code,
                    candidate_rank=c.candidate_rank,
                    station_id=c.station_id,
                    station_name=c.station_name,
                    station_latitude=c.station_latitude,
                    station_longitude=c.station_longitude,
                    elevation_m=c.elevation_m,
                    distance_km=c.distance_km,
                    network=c.network,
                    archive_begin=c.archive_begin,
                    archive_end=c.archive_end,
                    is_exact_code=c.is_exact_code,
                    is_icao_match=c.is_icao_match,
                    archive_covers_2025=c.archive_covers_2025,
                    is_selected=is_sel,
                )
            )

    # Write airport_station_mapping.csv
    mapping_file = output_dir / "airport_station_mapping.csv"
    with mapping_file.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "origin_airport_id",
            "airport_code",
            "airport_name",
            "airport_latitude",
            "airport_longitude",
            "station_id",
            "station_name",
            "station_latitude",
            "station_longitude",
            "elevation_m",
            "distance_km",
            "match_type",
            "selection_reason",
            "network",
            "archive_begin",
            "archive_end",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_mappings:
            writer.writerow(asdict(m))

    # Write airport_station_candidates.csv
    candidates_file = output_dir / "airport_station_candidates.csv"
    with candidates_file.open("w", encoding="utf-8", newline="") as f:
        cand_fieldnames = [
            "airport_id",
            "airport_code",
            "candidate_rank",
            "station_id",
            "station_name",
            "station_latitude",
            "station_longitude",
            "elevation_m",
            "distance_km",
            "network",
            "archive_begin",
            "archive_end",
            "is_exact_code",
            "is_icao_match",
            "archive_covers_2025",
            "is_selected",
        ]
        writer = csv.DictWriter(f, fieldnames=cand_fieldnames)
        writer.writeheader()
        for c in all_candidates:
            writer.writerow(asdict(c))

    # Compute summary statistics
    distances = [m.distance_km for m in all_mappings]
    sorted_dists = sorted(distances)
    n = len(sorted_dists)

    match_types: dict[str, int] = {}
    for m in all_mappings:
        match_types[m.match_type] = match_types.get(m.match_type, 0) + 1

    active_cov_count = sum(
        1 for m in all_mappings if is_active_during_2025(m.archive_begin, m.archive_end)
    )
    flagged_count = sum(
        1 for m in all_mappings if m.match_type == "FLAGGED_FOR_MANUAL_REVIEW"
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inventory_source": str(inventory_path),
        "inventory_records": len(airports),
        "inventory_flight_count": sum(a.flight_count for a in airports),
        "catalogue_source": str(catalogue_path),
        "catalogue_sha256": compute_sha256(catalogue_path),
        "catalogue_total_stations": len(stations),
        "total_airports_mapped": len(all_mappings),
        "match_type_counts": match_types,
        "distance_summary_km": {
            "min": round(sorted_dists[0], 3),
            "max": round(sorted_dists[-1], 3),
            "mean": round(sum(sorted_dists) / n, 3),
            "median": round(sorted_dists[n // 2], 3),
            "p95": round(sorted_dists[int(n * 0.95)], 3),
        },
        "on_airport_count_le_5km": sum(1 for d in distances if d <= 5.0),
        "active_2025_archive_count": active_cov_count,
        "flagged_review_count": flagged_count,
        "files_generated": [
            str(mapping_file),
            str(candidates_file),
            str(output_dir / "airport_station_mapping_summary.json"),
        ],
    }

    summary_file = output_dir / "airport_station_mapping_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

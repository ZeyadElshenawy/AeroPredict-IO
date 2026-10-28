"""Unit tests for weather station mapper, distance calculation, and ranking."""

from pathlib import Path

import pytest

from flightops_ml.ingestion.weather_station_mapper import (
    AirportRecord,
    WeatherStationRecord,
    find_candidate_stations,
    haversine_distance_km,
    load_airport_inventory,
    load_station_catalogue,
    map_airports_to_weather_stations,
    select_best_station,
)

# --- 1. Distance Calculations ---


def test_haversine_zero_distance_for_identical_points():
    dist = haversine_distance_km(32.8998, -97.0403, 32.8998, -97.0403)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_reference_distance():
    # JFK (40.6413, -73.7781) to LHR (51.4700, -0.4543) ~ 5555 km
    dist = haversine_distance_km(40.6413, -73.7781, 51.4700, -0.4543)
    assert 5540.0 <= dist <= 5570.0


def test_haversine_equator_and_poles():
    # North Pole to South Pole: half circumference ~ 20015 km
    dist = haversine_distance_km(90.0, 0.0, -90.0, 0.0)
    assert dist == pytest.approx(20015.0, rel=1e-3)

    # 1 degree along equator ~ 111.19 km
    dist_deg = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
    assert dist_deg == pytest.approx(111.19, rel=1e-2)


@pytest.mark.parametrize(
    ("lat1", "lon1", "lat2", "lon2"),
    [
        (91.0, 0.0, 0.0, 0.0),
        (-91.0, 0.0, 0.0, 0.0),
        (0.0, 181.0, 0.0, 0.0),
        (0.0, -181.0, 0.0, 0.0),
        (0.0, 0.0, 95.0, 0.0),
    ],
)
def test_haversine_rejects_out_of_bounds_coordinates(lat1, lon1, lat2, lon2):
    with pytest.raises(ValueError, match="out of range"):
        haversine_distance_km(lat1, lon1, lat2, lon2)


# --- 2. Catalogue Loading & Validation ---


def test_load_station_catalogue_validates_columns(tmp_path: Path):
    bad_csv = tmp_path / "bad_catalogue.csv"
    bad_csv.write_text("stid,station_name,lat\nDFW,Dallas,32.89\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_station_catalogue(bad_csv)


def test_load_station_catalogue_preserves_leading_zeros(tmp_path: Path):
    cat_csv = tmp_path / "cat.csv"
    cat_csv.write_text(
        "stid,station_name,lat,lon,elev,begints,endts,iem_network\n"
        "01M,Collierville,35.05,-89.67,100,2010-01-01 00:00,,TN_ASOS\n"
        "DFW,Dallas/Fort Worth,32.89,-97.04,180,1947-01-01 00:00,,TX_ASOS\n",
        encoding="utf-8",
    )
    stations = load_station_catalogue(cat_csv)
    assert len(stations) == 2
    assert stations[0].stid == "01M"  # Preserved leading zero
    assert stations[1].stid == "DFW"


def test_load_station_catalogue_rejects_duplicates(tmp_path: Path):
    cat_csv = tmp_path / "dupe_cat.csv"
    cat_csv.write_text(
        "stid,station_name,lat,lon,elev,begints,endts,iem_network\n"
        "DFW,Dallas 1,32.89,-97.04,180,1947-01-01 00:00,,TX_ASOS\n"
        "DFW,Dallas 2,32.90,-97.05,185,1950-01-01 00:00,,TX_ASOS\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate station identifier 'DFW'"):
        load_station_catalogue(cat_csv)


def test_load_station_catalogue_rejects_invalid_coordinates(tmp_path: Path):
    cat_csv = tmp_path / "invalid_coords.csv"
    cat_csv.write_text(
        "stid,station_name,lat,lon,elev,begints,endts,iem_network\n"
        "BAD,Bad Station,not_a_float,-97.04,180,1947-01-01 00:00,,TX_ASOS\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid float coordinates"):
        load_station_catalogue(cat_csv)


# --- 3. Airport Inventory Loading & Validation ---


def test_load_airport_inventory_validates_columns(tmp_path: Path):
    bad_csv = tmp_path / "bad_inventory.csv"
    bad_csv.write_text("origin_airport_id,latitude\n11298,32.89\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_airport_inventory(bad_csv)


def test_load_airport_inventory_rejects_duplicate_ids(tmp_path: Path):
    inv_csv = tmp_path / "dupe_inventory.csv"
    inv_csv.write_text(
        "origin_airport_id,latitude,longitude,flight_count,active_days\n"
        "11298,32.896,-97.038,1000,365\n"
        "11298,32.896,-97.038,500,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate origin_airport_id"):
        load_airport_inventory(inv_csv)


def test_load_airport_inventory_rejects_nonpositive_counts(tmp_path: Path):
    inv_csv = tmp_path / "bad_counts.csv"
    inv_csv.write_text(
        "origin_airport_id,latitude,longitude,flight_count,active_days\n"
        "11298,32.896,-97.038,0,365\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nonpositive flight count"):
        load_airport_inventory(inv_csv)


# --- 4. Candidate Ranking & Matching Logic ---


def make_test_station(
    stid: str,
    lat: float,
    lon: float,
    name: str = "Test Station",
    begints: str = "2000-01-01 00:00",
    endts: str = "",
    network: str = "TX_ASOS",
) -> WeatherStationRecord:
    return WeatherStationRecord(
        stid=stid,
        station_name=name,
        latitude=lat,
        longitude=lon,
        elevation_m=100.0,
        begints=begints,
        endts=endts,
        iem_network=network,
    )


def test_find_candidate_stations_ranks_by_distance():
    airport = AirportRecord(
        origin_airport_id=11298,
        airport_code="DFW",
        airport_name="Dallas/Fort Worth International",
        latitude=32.8968,
        longitude=-97.0380,
        flight_count=1000,
        active_days=365,
    )
    st_near = make_test_station("DFW", 32.8980, -97.0400, name="DFW Field")
    st_mid = make_test_station("DAL", 32.8471, -96.8517, name="Dallas Love")
    st_far = make_test_station("FTW", 32.8197, -97.3624, name="Meacham")

    cands = find_candidate_stations(
        airport, [st_far, st_near, st_mid], max_candidates=5
    )
    assert len(cands) == 3
    assert cands[0].station_id == "DFW"
    assert cands[0].candidate_rank == 1
    assert cands[1].station_id == "DAL"
    assert cands[1].candidate_rank == 2
    assert cands[2].station_id == "FTW"
    assert cands[2].candidate_rank == 3
    assert cands[0].distance_km < cands[1].distance_km < cands[2].distance_km


def test_select_best_station_prefers_exact_iata_on_airport():
    airport = AirportRecord(
        origin_airport_id=10397,
        airport_code="ATL",
        airport_name="Hartsfield-Jackson Atlanta International",
        latitude=33.6407,
        longitude=-84.4277,
        flight_count=5000,
        active_days=365,
    )
    st_atl = make_test_station("ATL", 33.6400, -84.4270, name="Atlanta Int AP")
    st_other = make_test_station("FTY", 33.7790, -84.5214, name="Fulton County")

    cands = find_candidate_stations(airport, [st_other, st_atl])
    best = select_best_station(airport, cands)

    assert best.station_id == "ATL"
    assert best.match_type == "EXACT_IATA_ON_AIRPORT"
    assert best.distance_km <= 1.0


def test_select_best_station_handles_territorial_icao_prefix():
    airport = AirportRecord(
        origin_airport_id=10299,
        airport_code="ANC",
        airport_name="Ted Stevens Anchorage International",
        latitude=61.1744,
        longitude=-149.9964,
        flight_count=2000,
        active_days=365,
    )
    st_panc = make_test_station(
        "PANC",
        61.1740,
        -149.9960,
        name="ANCHORAGE INTL ARPT (ASOS)",
        network="AK_ASOS",
    )
    cands = find_candidate_stations(airport, [st_panc])
    best = select_best_station(airport, cands)

    assert best.station_id == "PANC"
    assert best.match_type == "ICAO_PREFIX_ON_AIRPORT"
    assert best.distance_km <= 1.0


def test_select_best_station_handles_colocated_airfield_facility():
    # AZA with FAA LID IWA located 0.1 km away
    airport = AirportRecord(
        origin_airport_id=10466,
        airport_code="AZA",
        airport_name="Phoenix-Mesa Gateway",
        latitude=33.3078,
        longitude=-111.6555,
        flight_count=1000,
        active_days=365,
    )
    st_iwa = make_test_station(
        "IWA",
        33.3080,
        -111.6560,
        name="WILLIAMS AFB/CHANDL",
        network="AZ_ASOS",
    )
    cands = find_candidate_stations(airport, [st_iwa])
    best = select_best_station(airport, cands)

    assert best.station_id == "IWA"
    assert best.match_type == "AIRPORT_FACILITY_COORDINATE_MATCH"
    assert best.distance_km <= 1.0


def test_select_best_station_flags_distant_station_for_manual_review():
    # Remote airport with nearest station 45 km away
    airport = AirportRecord(
        origin_airport_id=99999,
        airport_code="RMT",
        airport_name="Remote Island Strip",
        latitude=25.0000,
        longitude=-70.0000,
        flight_count=50,
        active_days=50,
    )
    st_distant = make_test_station("FAR", 25.4000, -70.0000, name="Distant Station")

    cands = find_candidate_stations(airport, [st_distant], search_radius_km=100.0)
    best = select_best_station(
        airport,
        cands,
        on_airport_threshold_km=5.0,
        max_allowable_distance_km=30.0,
    )

    assert best.match_type == "FLAGGED_FOR_MANUAL_REVIEW"
    assert "No defensible candidate within 30.0 km" in best.selection_reason


def test_select_best_station_rejects_inactive_2025_archive_as_on_airport():
    airport = AirportRecord(
        origin_airport_id=10397,
        airport_code="ATL",
        airport_name="Atlanta Int",
        latitude=33.6407,
        longitude=-84.4277,
        flight_count=5000,
        active_days=365,
    )
    # Station decommissioned in 2024
    st_defunct = make_test_station("ATL", 33.6407, -84.4277, endts="2024-06-01 00:00")
    # Active station nearby (8 km)
    st_backup = make_test_station("FTY", 33.7790, -84.5214, endts="")

    cands = find_candidate_stations(airport, [st_defunct, st_backup])
    best = select_best_station(airport, cands)

    # Defunct station should not be chosen; backup chosen as fallback
    assert best.station_id == "FTY"
    assert best.match_type == "NEARBY_FALLBACK_VICINITY"


# --- 5. Deterministic End-to-End Pipeline Output ---


def test_map_airports_deterministic_output(tmp_path: Path):
    inv_file = tmp_path / "inventory.csv"
    inv_file.write_text(
        "origin_airport_id,latitude,longitude,flight_count,active_days\n"
        "10397,33.6407,-84.4277,5000,365\n"
        "11298,32.8968,-97.0380,4000,365\n",
        encoding="utf-8",
    )
    cat_file = tmp_path / "catalogue.csv"
    cat_file.write_text(
        "stid,station_name,lat,lon,elev,begints,endts,iem_network\n"
        "ATL,Atlanta Int,33.6400,-84.4270,300,1940-01-01 00:00,,GA_ASOS\n"
        "DFW,Dallas Fort Worth,32.8980,-97.0400,180,1947-01-01 00:00,,TX_ASOS\n",
        encoding="utf-8",
    )
    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"

    res1 = map_airports_to_weather_stations(inv_file, cat_file, out_dir_1)
    res2 = map_airports_to_weather_stations(inv_file, cat_file, out_dir_2)

    assert res1["total_airports_mapped"] == 2
    assert res1["match_type_counts"] == res2["match_type_counts"]
    assert res1["distance_summary_km"] == res2["distance_summary_km"]

    csv1 = (out_dir_1 / "airport_station_mapping.csv").read_text(encoding="utf-8")
    csv2 = (out_dir_2 / "airport_station_mapping.csv").read_text(encoding="utf-8")
    assert csv1 == csv2

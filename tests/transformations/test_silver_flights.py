from datetime import date, datetime, time
from unittest.mock import MagicMock

import pytest
from scripts import transform_silver_flights as orchestrator

from flightops_ml.transformations import silver_flights as sf


def test_parse_flight_date_us_format():
    assert sf.parse_flight_date("7/24/2025 12:00:00 AM") == date(2025, 7, 24)
    assert sf.parse_flight_date("01/05/2025 12:00:00 AM") == date(2025, 1, 5)
    assert sf.parse_flight_date("12/31/2025") == date(2025, 12, 31)


def test_parse_flight_date_iso_format():
    assert sf.parse_flight_date("2025-07-24") == date(2025, 7, 24)
    assert sf.parse_flight_date("2025-01-05 00:00:00") == date(2025, 1, 5)


def test_parse_flight_date_rejects_malformed_dates():
    with pytest.raises(ValueError, match="Unrecognized flight date format"):
        sf.parse_flight_date("notadate")
    with pytest.raises(ValueError, match="Invalid ISO flight date"):
        sf.parse_flight_date("not-a-date")
    with pytest.raises(ValueError, match="Invalid flight date values"):
        sf.parse_flight_date("02/30/2025 12:00:00 AM")
    with pytest.raises(ValueError, match="Invalid flight date values"):
        sf.parse_flight_date("13/01/2025 12:00:00 AM")


def test_normalize_hhmm_time():
    assert sf.normalize_hhmm_time("600") == "0600"
    assert sf.normalize_hhmm_time("0922") == "0922"
    assert sf.normalize_hhmm_time("0000") == "0000"
    assert sf.normalize_hhmm_time("2400") == "2400"


def test_normalize_hhmm_time_rejects_invalid_values():
    with pytest.raises(ValueError, match="numeric format"):
        sf.normalize_hhmm_time("abcd")
    with pytest.raises(ValueError, match="Out of range"):
        sf.normalize_hhmm_time("2500")
    with pytest.raises(ValueError, match="Out of range"):
        sf.normalize_hhmm_time("0960")
    with pytest.raises(ValueError, match="24:xx is invalid"):
        sf.normalize_hhmm_time("2405")


def test_parse_clock_time():
    assert sf.parse_clock_time("0922") == time(9, 22, 0)
    assert sf.parse_clock_time("0000") == time(0, 0, 0)
    assert sf.parse_clock_time("2400") == time(0, 0, 0)


def test_build_crs_dep_timestamp_normal():
    flight_date = date(2025, 7, 24)
    expected = datetime(2025, 7, 24, 9, 22, 0)
    assert sf.build_crs_dep_timestamp(flight_date, "0922") == expected


def test_build_crs_dep_timestamp_2400_rolls_to_following_day_midnight():
    flight_date = date(2025, 7, 24)
    expected = datetime(2025, 7, 25, 0, 0, 0)
    assert sf.build_crs_dep_timestamp(flight_date, "2400") == expected

    # Month boundary test
    month_end = date(2025, 1, 31)
    expected_feb = datetime(2025, 2, 1, 0, 0, 0)
    assert sf.build_crs_dep_timestamp(month_end, "2400") == expected_feb


def test_clean_flag():
    assert sf.clean_flag("1.00", "CANCELLED") is True
    assert sf.clean_flag("0.00", "CANCELLED") is False
    assert sf.clean_flag("1", "DIVERTED") is True
    assert sf.clean_flag("0", "DIVERTED") is False
    assert sf.clean_flag("Y", "FLAG") is True
    assert sf.clean_flag("N", "FLAG") is False

    with pytest.raises(ValueError, match="Missing required flag"):
        sf.clean_flag(None, "CANCELLED")
    with pytest.raises(ValueError, match="Invalid flag value"):
        sf.clean_flag("2.00", "CANCELLED")


def test_clean_dep_del15_for_cancelled_flights():
    # Requirement 7: Preserve dep_del15 as NULL for cancelled flights; never impute to 0
    assert sf.clean_dep_del15("1.00", cancelled=True) is None
    assert sf.clean_dep_del15("0.00", cancelled=True) is None
    assert sf.clean_dep_del15("", cancelled=True) is None
    assert sf.clean_dep_del15(None, cancelled=True) is None


def test_clean_dep_del15_for_non_cancelled_flights():
    assert sf.clean_dep_del15("0.00", cancelled=False) == 0
    assert sf.clean_dep_del15("1.00", cancelled=False) == 1
    assert sf.clean_dep_del15("", cancelled=False) is None
    assert sf.clean_dep_del15(None, cancelled=False) is None

    with pytest.raises(ValueError, match="Unexpected DEP_DEL15 label"):
        sf.clean_dep_del15("2.00", cancelled=False)


def test_compute_training_eligibility():
    # Valid label, not cancelled, not diverted -> True
    assert sf.compute_training_eligibility(1, cancelled=False, diverted=False) is True
    assert sf.compute_training_eligibility(0, cancelled=False, diverted=False) is True

    # Cancelled -> False
    assert (
        sf.compute_training_eligibility(None, cancelled=True, diverted=False) is False
    )
    assert sf.compute_training_eligibility(1, cancelled=True, diverted=False) is False

    # Diverted -> False
    assert sf.compute_training_eligibility(0, cancelled=False, diverted=True) is False
    assert sf.compute_training_eligibility(1, cancelled=False, diverted=True) is False

    # Missing label -> False
    assert (
        sf.compute_training_eligibility(None, cancelled=False, diverted=False) is False
    )


def test_parse_elapsed_time_and_distance():
    assert sf.parse_elapsed_time("98.00") == 98.0
    assert sf.parse_elapsed_time("") is None
    assert sf.parse_elapsed_time(None) is None

    # Negative values can occur due to clerical/timezone anomalies in raw BTS data
    assert sf.parse_elapsed_time("-10.0") == -10.0

    assert sf.parse_distance("372.00") == 372.0
    with pytest.raises(ValueError, match="Missing required DISTANCE"):
        sf.parse_distance("")
    with pytest.raises(ValueError, match="Negative DISTANCE"):
        sf.parse_distance("-50.0")


def test_transform_flight_row_success():
    raw = {
        "source_sha256": "a" * 64,
        "source_row_number": "100",
        "ingestion_run_id": "1",
        "ingested_at": datetime(2025, 2, 1, 10, 0),
        "flight_date_raw": "7/24/2025 12:00:00 AM",
        "year_raw": "2025",
        "quarter_raw": "3",
        "month_raw": "7",
        "day_of_month_raw": "24",
        "day_of_week_raw": "4",
        "op_unique_carrier_raw": "DL",
        "origin_airport_id_raw": "10397",
        "origin_raw": "ATL",
        "dest_airport_id_raw": "12892",
        "dest_raw": "LAX",
        "crs_dep_time_raw": "2400",
        "crs_arr_time_raw": "0215",
        "crs_elapsed_time_raw": "255.00",
        "distance_raw": "1947.00",
        "cancelled_raw": "0.00",
        "diverted_raw": "0.00",
        "dep_del15_raw": "1.00",
    }

    result = sf.transform_flight_row(raw)

    assert result["flight_date"] == date(2025, 7, 24)
    assert result["crs_dep_time_raw"] == "2400"
    assert result["crs_dep_time_str"] == "2400"
    assert result["crs_dep_time"] == time(0, 0, 0)
    assert result["crs_dep_timestamp"] == datetime(2025, 7, 25, 0, 0, 0)
    assert result["cancelled"] is False
    assert result["diverted"] is False
    assert result["dep_del15"] == 1
    assert result["is_training_eligible"] is True
    assert result["crs_elapsed_time"] == 255.0
    assert result["distance"] == 1947.0


def test_transform_flight_row_calendar_mismatch_fails():
    raw = {
        "source_sha256": "a" * 64,
        "source_row_number": "1",
        "ingestion_run_id": "1",
        "ingested_at": datetime.now(),
        "flight_date_raw": "7/24/2025 12:00:00 AM",
        "year_raw": "2025",
        "quarter_raw": "3",
        "month_raw": "8",  # Mismatched month!
        "day_of_month_raw": "24",
        "day_of_week_raw": "4",
        "op_unique_carrier_raw": "DL",
        "origin_airport_id_raw": "10397",
        "origin_raw": "ATL",
        "dest_airport_id_raw": "12892",
        "dest_raw": "LAX",
        "crs_dep_time_raw": "1200",
        "crs_arr_time_raw": "1400",
        "crs_elapsed_time_raw": "120.00",
        "distance_raw": "500.00",
        "cancelled_raw": "0.00",
        "diverted_raw": "0.00",
        "dep_del15_raw": "0.00",
    }
    with pytest.raises(ValueError, match="Calendar field mismatch"):
        sf.transform_flight_row(raw)


def test_transform_flight_row_airport_code_length_fails():
    raw = {
        "source_sha256": "a" * 64,
        "source_row_number": "1",
        "ingestion_run_id": "1",
        "ingested_at": datetime.now(),
        "flight_date_raw": "7/24/2025 12:00:00 AM",
        "year_raw": "2025",
        "quarter_raw": "3",
        "month_raw": "7",
        "day_of_month_raw": "24",
        "day_of_week_raw": "4",
        "op_unique_carrier_raw": "DL",
        "origin_airport_id_raw": "10397",
        "origin_raw": "INVALID",
        "dest_airport_id_raw": "12892",
        "dest_raw": "LAX",
        "crs_dep_time_raw": "1200",
        "crs_arr_time_raw": "1400",
        "crs_elapsed_time_raw": "120.00",
        "distance_raw": "500.00",
        "cancelled_raw": "0.00",
        "diverted_raw": "0.00",
        "dep_del15_raw": "0.00",
    }
    with pytest.raises(ValueError, match="Invalid airport code length"):
        sf.transform_flight_row(raw)


def test_validate_bronze_partition_raises_on_violations():
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value
    # mock counts: total=100, bad_dates=2, other bad fields=0
    fake_cur.fetchone.return_value = (100, 2, 0, 0, 0, 0, 0, 0)
    fake_cur.fetchall.return_value = [(1, "bad-date-1"), (2, "bad-date-2")]

    with pytest.raises(ValueError, match="failed validation with 2 violations"):
        sf.validate_bronze_partition(fake_conn, "test-checksum")


def test_get_silver_state_incomplete():
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value
    # bronze=100, silver=99, min_row=1, max_row=99
    fake_cur.fetchone.side_effect = [(100,), (99, 1, 99)]

    bronze, silver, is_complete = sf.get_silver_state(fake_conn, "checksum")
    assert bronze == 100
    assert silver == 99
    assert is_complete is False


def test_get_silver_state_complete():
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value
    # bronze=100, silver=100, min_row=1, max_row=100
    fake_cur.fetchone.side_effect = [(100,), (100, 1, 100)]

    bronze, silver, is_complete = sf.get_silver_state(fake_conn, "checksum")
    assert bronze == 100
    assert silver == 100
    assert is_complete is True


def test_orchestrator_process_archive_dry_run():
    fake_conn = MagicMock()
    fake_cur = fake_conn.cursor.return_value.__enter__.return_value
    fake_cur.fetchone.side_effect = [(100,), (0, None, None)]

    status = orchestrator.process_archive(
        fake_conn,
        year=2025,
        month=1,
        checksum="test-checksum",
        filename="2025-01.zip",
        execute=False,
    )
    assert status == "planned"

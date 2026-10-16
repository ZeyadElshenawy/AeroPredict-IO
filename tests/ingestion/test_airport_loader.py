import csv
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from flightops_ml.ingestion.airport_loader import (
    EXPECTED_COLUMNS,
    load_airport_reference,
    optional_coordinate,
    optional_date,
    required_flag,
)


def write_airport_csv(path: Path, rows: list[dict] | None = None) -> Path:
    if rows is None:
        rows = [
            {
                "AIRPORT_SEQ_ID": "1039708",
                "AIRPORT_ID": "10397",
                "AIRPORT": "ATL",
                "DISPLAY_AIRPORT_NAME": "Hartsfield-Jackson Atlanta International",
                "LATITUDE": "33.6407",
                "LONGITUDE": "-84.4277",
                "UTC_LOCAL_TIME_VARIATION": "-0500",
                "AIRPORT_START_DATE": "2000-01-01",
                "AIRPORT_THRU_DATE": "",
                "AIRPORT_IS_CLOSED": "N",
                "AIRPORT_IS_LATEST": "Y",
            }
        ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_optional_coordinate_accepts_blank_and_valid_values():
    assert optional_coordinate("", "LATITUDE", 1, -90, 90) is None
    assert optional_coordinate("33.6407", "LATITUDE", 1, -90, 90) == 33.6407


@pytest.mark.parametrize("value", ["north", "91", "-91"])
def test_optional_coordinate_rejects_invalid_latitude(value):
    with pytest.raises(ValueError):
        optional_coordinate(value, "LATITUDE", 1, -90, 90)


def test_optional_date_accepts_blank_and_iso_date():
    assert optional_date("", "AIRPORT_THRU_DATE", 1) is None
    assert optional_date("2025-01-31", "AIRPORT_START_DATE", 1) == date(2025, 1, 31)


def test_required_flag_accepts_y_and_n():
    assert required_flag("Y", "AIRPORT_IS_LATEST", 1) is True
    assert required_flag("N", "AIRPORT_IS_LATEST", 1) is False


def test_required_flag_rejects_unknown_value():
    with pytest.raises(ValueError, match="expected Y, N, 1, or 0"):
        required_flag("maybe", "AIRPORT_IS_LATEST", 1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Y", True),
        ("N", False),
        ("1", True),
        ("0", False),
    ],
)
def test_required_flag_accepts_supported_values(value, expected):
    assert required_flag(value, "AIRPORT_IS_CLOSED", 1) is expected


def test_load_airport_reference_rejects_wrong_schema_before_database_access(
    tmp_path,
):
    path = tmp_path / "airports.csv"
    path.write_text("AIRPORT_ID,AIRPORT\n10397,ATL\n", encoding="utf-8")

    with patch(
        "flightops_ml.ingestion.airport_loader.database_connection"
    ) as database_connection:
        with pytest.raises(ValueError, match="CSV columns"):
            load_airport_reference(path)

    database_connection.assert_not_called()


def test_load_airport_reference_rejects_duplicate_sequence_ids_before_database_access(
    tmp_path,
):
    row = {
        "AIRPORT_SEQ_ID": "1039708",
        "AIRPORT_ID": "10397",
        "AIRPORT": "ATL",
        "DISPLAY_AIRPORT_NAME": "Atlanta",
        "LATITUDE": "33.6407",
        "LONGITUDE": "-84.4277",
        "UTC_LOCAL_TIME_VARIATION": "-0500",
        "AIRPORT_START_DATE": "2000-01-01",
        "AIRPORT_THRU_DATE": "",
        "AIRPORT_IS_CLOSED": "N",
        "AIRPORT_IS_LATEST": "Y",
    }
    path = write_airport_csv(tmp_path / "airports.csv", [row, row.copy()])

    with patch(
        "flightops_ml.ingestion.airport_loader.database_connection"
    ) as database_connection:
        with pytest.raises(ValueError, match="Duplicate AIRPORT_SEQ_ID"):
            load_airport_reference(path)

    database_connection.assert_not_called()


def test_load_airport_reference_rejects_only_one_coordinate_before_database_access(
    tmp_path,
):
    row = {
        "AIRPORT_SEQ_ID": "1039708",
        "AIRPORT_ID": "10397",
        "AIRPORT": "ATL",
        "DISPLAY_AIRPORT_NAME": "Atlanta",
        "LATITUDE": "33.6407",
        "LONGITUDE": "",
        "UTC_LOCAL_TIME_VARIATION": "-0500",
        "AIRPORT_START_DATE": "2000-01-01",
        "AIRPORT_THRU_DATE": "",
        "AIRPORT_IS_CLOSED": "N",
        "AIRPORT_IS_LATEST": "Y",
    }
    path = write_airport_csv(tmp_path / "airports.csv", [row])

    with patch(
        "flightops_ml.ingestion.airport_loader.database_connection"
    ) as database_connection:
        with pytest.raises(ValueError, match="both be present or both be blank"):
            load_airport_reference(path)

    database_connection.assert_not_called()

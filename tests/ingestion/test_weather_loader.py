from datetime import datetime
from pathlib import Path

import pytest

from flightops_ml.ingestion.weather_loader import (
    clean_value,
    load_weather_file,
    parse_observation_time,
)


def test_parse_observation_time_interprets_timestamp_as_utc():
    result = parse_observation_time("2025-01-01 00:53")

    # The database column stores a naive timestamp, by design.
    # Its documented interpretation is UTC.
    assert result == datetime(2025, 1, 1, 0, 53)


def test_parse_observation_time_rejects_invalid_timestamp():
    with pytest.raises(ValueError):
        parse_observation_time("not-a-timestamp")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("M", None),
        ("", None),
        ("   ", None),
        (None, None),
        ("51.00", "51.00"),
        ("T", "T"),
        ("CLR", "CLR"),
    ],
)
def test_clean_value_preserves_source_values(value, expected):
    assert clean_value(value) == expected


def test_load_weather_file_rejects_wrong_csv_schema(tmp_path: Path):
    csv_path = tmp_path / "DFW.csv"
    csv_path.write_text(
        "station,valid,tmpf\nDFW,2025-01-01 00:00,51.00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV columns"):
        load_weather_file(csv_path, year=2025, month=1)

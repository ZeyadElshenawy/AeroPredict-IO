"""Unit tests for the bulk weather ingestion module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flightops_ml.ingestion.weather_bulk_ingester import (
    WEATHER_REQUEST_FIELDS,
    build_iem_batch_params,
    clean_weather_value,
    compute_file_sha256,
    fetch_iem_csv_stream,
    ingest_weather_batch,
    parse_and_deduplicate_batch_rows,
)


def test_build_iem_batch_params_validates_month_boundaries():
    params_jan = build_iem_batch_params(["DFW", "ATL"], year=2025, month=1)
    assert params_jan["year1"] == "2025"
    assert params_jan["month1"] == "1"
    assert params_jan["day1"] == "1"
    assert params_jan["year2"] == "2025"
    assert params_jan["month2"] == "1"
    assert params_jan["day2"] == "31"
    assert params_jan["hour2"] == "23"
    assert params_jan["minute2"] == "59"
    assert params_jan["station"] == ["ATL", "DFW"]
    assert "report_type" in params_jan
    assert set(params_jan["data"]) == set(WEATHER_REQUEST_FIELDS)

    # February in non-leap year (2025) has 28 days
    params_feb = build_iem_batch_params(["DFW"], year=2025, month=2)
    assert params_feb["day2"] == "28"

    # Annual mode (month=None) covers full year
    params_annual = build_iem_batch_params(["DFW"], year=2025, month=None)
    assert params_annual["month1"] == "1"
    assert params_annual["month2"] == "12"
    assert params_annual["day2"] == "31"

    # Invalid month raises ValueError
    with pytest.raises(ValueError, match="Month must be between 1 and 12"):
        build_iem_batch_params(["DFW"], year=2025, month=13)


def test_clean_weather_value_mappings():
    assert clean_weather_value("50.00") == "50.00"
    assert clean_weather_value("M") is None
    assert clean_weather_value("m") is None
    assert clean_weather_value("") is None
    assert clean_weather_value("   ") is None
    assert clean_weather_value(None) is None
    assert clean_weather_value("CLR") == "CLR"


def test_parse_and_deduplicate_batch_rows_prioritizes_richer_report():
    sample_csv = "\n".join(
        [
            "# Some IEM comment header",
            (
                "station,valid,tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,"
                "skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,metar"
            ),
            (
                "PAWG,2025-01-09 00:55,42.80,35.00,70.00,120.00,7.00,0.00,30.39,"
                "1029.00,10.00,M,SCT,BKN,OVC,M,2400,4300,5000,M,M,"
                "PAWG 090055Z 12007KT 10SM SCT024 BKN043 OVC050"
            ),
            (
                "PAWG,2025-01-09 00:55,M,M,M,120.00,5.00,M,30.39,M,10.00,M,M,M,M,"
                "M,M,M,M,M,M,PAWG 090055Z AUTO 12005KT 10SM"
            ),
            (
                "01M,2025-01-09 01:00,50.00,30.00,45.00,180.00,10.00,0.00,30.00,"
                "1015.00,10.00,M,CLR,M,M,M,M,M,M,M,M,K01M 090100Z AUTO"
            ),
            "",
        ]
    )
    rows = parse_and_deduplicate_batch_rows(
        csv_text=sample_csv,
        source_sha256="test_sha",
        target_year=2025,
        target_month=1,
    )

    # Exactly 2 deduplicated records: 1 for PAWG at 00:55, 1 for 01M at 01:00
    assert len(rows) == 2

    # Verify both records were extracted and deduplicated properly
    row_map = {r[0]: r for r in rows}
    assert "PAWG" in row_map
    assert "01M" in row_map

    # Verify PAWG retained richer report (tmpf=42.80, not None)
    pawg_row = row_map["PAWG"]
    assert pawg_row[1] == datetime(2025, 1, 9, 0, 55)
    assert pawg_row[2] == "test_sha"
    # tmpf is the 5th element in output tuple
    assert pawg_row[4] == "42.80"
    assert pawg_row[5] == "35.00"

    # Verify leading-zero station ID preservation
    zero_row = row_map["01M"]
    assert zero_row[1] == datetime(2025, 1, 9, 1, 0)


def test_fetch_iem_csv_stream_retries_on_rate_limit():
    mock_session = MagicMock()
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = "station,valid\nDFW,2025-01-01 00:00\n"

    mock_session.get.side_effect = [resp_429, resp_200]

    result = fetch_iem_csv_stream(
        session=mock_session,
        stations=["DFW"],
        year=2025,
        month=1,
        max_retries=2,
        backoff_factor=0.01,
    )

    assert "DFW,2025-01-01 00:00" in result
    assert mock_session.get.call_count == 2


def test_fetch_iem_csv_stream_raises_after_max_retries():
    mock_session = MagicMock()
    resp_503 = MagicMock()
    resp_503.status_code = 503

    mock_session.get.return_value = resp_503

    with pytest.raises(RuntimeError, match="Failed to fetch batch"):
        fetch_iem_csv_stream(
            session=mock_session,
            stations=["DFW"],
            year=2025,
            month=1,
            max_retries=2,
            backoff_factor=0.01,
        )


def test_ingest_weather_batch_dry_run(tmp_path: Path):
    raw_file = tmp_path / "raw_weather.csv"
    sample_csv = "station,valid,tmpf\nDFW,2025-01-01 00:00,50.00\n"

    res = ingest_weather_batch(
        csv_text=sample_csv,
        raw_csv_path=raw_file,
        stations=["DFW"],
        year=2025,
        month=1,
        batch_id="test_batch",
        dry_run=True,
    )

    assert res.status == "dry_run"
    assert res.records_seen == 1
    assert res.records_inserted == 0
    assert raw_file.exists()
    assert compute_file_sha256(raw_file) == res.source_sha256

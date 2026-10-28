"""Unit tests for weather observation availability auditor and metrics calculation."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flightops_ml.ingestion.weather_auditor import (
    MonthlyStationAudit,
    StationAuditResult,
    build_iem_request_params,
    calculate_station_metrics,
    export_audit_artifacts,
    fetch_iem_batch,
    parse_observation_rows,
)


def test_build_iem_request_params_structure():
    params = build_iem_request_params(["DFW", "01M"], year=2025)
    assert params["station"] == ["DFW", "01M"]
    assert params["year1"] == 2025
    assert params["month1"] == 1
    assert params["day1"] == 1
    assert params["hour1"] == 0
    assert params["minute1"] == 0
    assert params["year2"] == 2025
    assert params["month2"] == 12
    assert params["day2"] == 31
    assert params["hour2"] == 23
    assert params["minute2"] == 59
    assert params["tz"] == "Etc/UTC"
    assert params["format"] == "onlycomma"
    assert params["missing"] == "M"
    assert params["report_type"] == ["3", "4"]


def test_parse_observation_rows_preserves_leading_zeros():
    csv_text = (
        "station,valid,tmpf,dwpf,sknt,drct,vsby,alti\n"
        "01M,2025-01-01 00:53,45.00,30.00,5.00,180.00,10.00,30.15\n"
        "DFW,2025-01-01 00:53,50.00,34.00,7.00,360.00,10.00,30.12\n"
    )
    parsed = parse_observation_rows(csv_text)
    assert "01M" in parsed
    assert "DFW" in parsed
    assert len(parsed["01M"]) == 1
    assert parsed["01M"][0]["timestamp"] == datetime(2025, 1, 1, 0, 53, tzinfo=UTC)


def test_parse_observation_rows_maps_missing_markers():
    csv_text = (
        "station,valid,tmpf,dwpf,sknt,drct,vsby,alti\n"
        "DFW,2025-01-01 00:53,M, ,7.00,M,10.00,30.12\n"
    )
    parsed = parse_observation_rows(csv_text)
    row = parsed["DFW"][0]
    assert row["tmpf"] is None
    assert row["dwpf"] is None
    assert row["sknt"] == "7.00"
    assert row["drct"] is None
    assert row["vsby"] == "10.00"
    assert row["alti"] == "30.12"


def test_calculate_station_metrics_annual_and_monthly_reconciliation():
    # Construct 10 synthetic observations across January and February
    obs = []
    # 5 obs in January
    for day in range(1, 6):
        obs.append(
            {
                "timestamp": datetime(2025, 1, day, 12, 0, tzinfo=UTC),
                "timestamp_str": f"2025-01-0{day} 12:00",
                "tmpf": "50.00",
                "dwpf": "30.00",
                "sknt": "5.00",
                "drct": "180.00",
                "vsby": "10.00",
                "alti": "30.00",
            }
        )
    # 5 obs in February (with 1 duplicate timestamp and 1 missing temp)
    for day in range(1, 5):
        obs.append(
            {
                "timestamp": datetime(2025, 2, day, 12, 0, tzinfo=UTC),
                "timestamp_str": f"2025-02-0{day} 12:00",
                "tmpf": None if day == 1 else "55.00",
                "dwpf": "35.00",
                "sknt": "6.00",
                "drct": "200.00",
                "vsby": "10.00",
                "alti": "29.95",
            }
        )
    # Add duplicate timestamp on Feb 4
    obs.append(
        {
            "timestamp": datetime(2025, 2, 4, 12, 0, tzinfo=UTC),
            "timestamp_str": "2025-02-04 12:00",
            "tmpf": "56.00",
            "dwpf": "35.00",
            "sknt": "7.00",
            "drct": "210.00",
            "vsby": "10.00",
            "alti": "29.95",
        }
    )

    annual, monthly = calculate_station_metrics(
        station_id="TEST",
        station_name="Test Airport",
        network="TX_ASOS",
        origin_airport_id=10001,
        airport_code="TST",
        observations=obs,
    )

    assert annual.total_observations == 10
    assert annual.distinct_timestamps == 9
    assert annual.duplicate_timestamps == 1
    assert annual.first_observation_utc == "2025-01-01 12:00"
    assert annual.last_observation_utc == "2025-02-04 12:00"

    # Monthly sum must exactly equal annual total
    monthly_sum_obs = sum(m.total_observations for m in monthly)
    assert monthly_sum_obs == annual.total_observations

    # January has 5, February has 5, March-December have 0
    jan = next(m for m in monthly if m.month == 1)
    feb = next(m for m in monthly if m.month == 2)
    mar = next(m for m in monthly if m.month == 3)

    assert jan.total_observations == 5
    assert jan.distinct_timestamps == 5
    assert feb.total_observations == 5
    assert feb.distinct_timestamps == 4
    assert mar.total_observations == 0


def test_calculate_station_metrics_gap_tracking():
    # Observations with a 10-hour gap
    obs = [
        {
            "timestamp": datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
            "timestamp_str": "2025-01-01 00:00",
            "tmpf": "50.00",
            "dwpf": "30.00",
            "sknt": "5.00",
            "drct": "180.00",
            "vsby": "10.00",
            "alti": "30.00",
        },
        {
            "timestamp": datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            "timestamp_str": "2025-01-01 10:00",
            "tmpf": "52.00",
            "dwpf": "31.00",
            "sknt": "6.00",
            "drct": "190.00",
            "vsby": "10.00",
            "alti": "30.05",
        },
    ]

    annual, _ = calculate_station_metrics(
        station_id="TEST",
        station_name="Test Airport",
        network="TX_ASOS",
        origin_airport_id=10001,
        airport_code="TST",
        observations=obs,
    )

    # Gap between 00:00 and 10:00 is 10.0 hours (>3h and >6h)
    assert annual.total_gaps_gt_3h >= 1
    assert annual.total_gaps_gt_6h >= 1
    assert annual.max_gap_hours >= 10.0


def test_calculate_station_metrics_zero_observations_handling():
    annual, monthly = calculate_station_metrics(
        station_id="DEAD",
        station_name="Offline Station",
        network="XX_ASOS",
        origin_airport_id=99999,
        airport_code="DED",
        observations=[],
    )

    assert annual.status == "FAILED"
    assert annual.total_observations == 0
    assert annual.distinct_timestamps == 0
    assert annual.requires_investigation is True
    assert annual.max_gap_hours == 8760.0
    assert annual.null_rate_tmpf == 1.0

    assert len(monthly) == 12
    for m in monthly:
        assert m.total_observations == 0
        assert m.completeness_pct == 0.0


def test_fetch_iem_batch_retries_on_rate_limit():
    mock_session = MagicMock()
    # First response: 429 Too Many Requests, Second response: 200 OK
    resp_429 = MagicMock()
    resp_429.status_code = 429

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = "station,valid,tmpf\nDFW,2025-01-01 00:53,50.00\n"

    mock_session.get.side_effect = [resp_429, resp_200]

    result = fetch_iem_batch(
        session=mock_session,
        stations=["DFW"],
        max_retries=2,
        backoff_factor=0.01,  # fast test
    )

    assert "DFW,2025-01-01 00:53,50.00" in result
    assert mock_session.get.call_count == 2


def test_fetch_iem_batch_raises_after_max_retries():
    mock_session = MagicMock()
    resp_503 = MagicMock()
    resp_503.status_code = 503

    mock_session.get.return_value = resp_503

    with pytest.raises(RuntimeError, match="Failed to fetch batch"):
        fetch_iem_batch(
            session=mock_session,
            stations=["DFW"],
            max_retries=2,
            backoff_factor=0.01,
        )


def test_export_audit_artifacts_deterministic(tmp_path: Path):
    annual_res = [
        StationAuditResult(
            station_id="DFW",
            station_name="Dallas Fort Worth",
            network="TX_ASOS",
            origin_airport_id=11298,
            airport_code="DFW",
            expected_period="2025-01-01 00:00 to 2025-12-31 23:59 UTC",
            expected_nominal_hours=8760,
            first_observation_utc="2025-01-01 00:00",
            last_observation_utc="2025-12-31 23:53",
            total_observations=8750,
            distinct_timestamps=8750,
            duplicate_timestamps=0,
            total_gaps_gt_3h=2,
            total_gaps_gt_6h=0,
            max_gap_hours=3.5,
            null_rate_tmpf=0.001,
            null_rate_dwpf=0.001,
            null_rate_sknt=0.0,
            null_rate_drct=0.0,
            null_rate_vsby=0.0,
            null_rate_alti=0.001,
            status="SUCCEEDED",
            requires_investigation=False,
            investigation_reason=None,
            error_message=None,
        )
    ]
    monthly_res = [
        MonthlyStationAudit(
            station_id="DFW",
            station_name="Dallas Fort Worth",
            year=2025,
            month=1,
            month_label="2025-01",
            nominal_hours=744,
            total_observations=743,
            distinct_timestamps=743,
            duplicate_timestamps=0,
            max_gap_hours=2.0,
            gaps_gt_3h=0,
            gaps_gt_6h=0,
            null_rate_tmpf=0.0,
            null_rate_dwpf=0.0,
            null_rate_sknt=0.0,
            null_rate_drct=0.0,
            null_rate_vsby=0.0,
            null_rate_alti=0.0,
            completeness_pct=99.87,
        )
    ]

    out_dir = tmp_path / "audit_out"
    summary = export_audit_artifacts(
        annual_results=annual_res,
        monthly_results=monthly_res,
        output_dir=out_dir,
        scope="test",
    )

    assert summary["stations_audited"] == 1
    assert summary["stations_succeeded"] == 1
    assert summary["stations_requiring_investigation"] == 0

    ann_csv = out_dir / "weather_availability_annual_test.csv"
    mon_csv = out_dir / "weather_availability_monthly_test.csv"
    sum_json = out_dir / "weather_availability_summary_test.json"

    assert ann_csv.exists()
    assert mon_csv.exists()
    assert sum_json.exists()

    with sum_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["aggregate_observations"] == 8750

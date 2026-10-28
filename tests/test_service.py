"""Unit and integration tests for FlightExplainer and FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from flightops_ml.explanation.explainer import FlightExplainer
from flightops_ml.service.app import app


@pytest.fixture(scope="module")
def explainer() -> FlightExplainer:
    return FlightExplainer(
        model_path="models/lgb_champion.txt",
        metadata_path="models/model_metadata.json",
    )


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_explainer_high_risk_flight(explainer: FlightExplainer) -> None:
    """Test that a high-risk flight triggers the operational alert."""
    high_risk_flight = {
        "op_unique_carrier": "UA",
        "origin": "ORD",
        "dest": "LGA",
        "dep_hour": 19,
        "day_of_week": 5,
        "month": 12,
        "distance": 733.0,
        "crs_elapsed_time_clean": 140.0,
        "weather_temp_f": 30.0,
        "weather_dew_point_f": 28.0,
        "weather_wind_speed_kt": 22.0,
        "weather_wind_gust_kt": 38.0,
        "weather_visibility_sm": 2.0,
        "weather_altimeter_in_hg": 29.50,
        "weather_has_snow": 1,
        "weather_has_precip": 1,
        "origin_hourly_flights": 48.0,
        "carrier_hub_daily_flights": 180.0,
        "hub_daily_total_flights": 380.0,
    }

    res = explainer.explain_flight(high_risk_flight, top_n=3)

    assert res["predicted_delay_probability"] >= 0.21
    assert res["alert_triggered"] is True
    assert res["risk_tier"] in ["HIGH", "SEVERE"]
    assert len(res["top_delay_reasons"]) == 3
    for reason in res["top_delay_reasons"]:
        assert "display_name" in reason
        assert "description" in reason
        assert reason["shap_impact"] > 0
        assert reason["direction"] == "RISK_INCREASING"


def test_explainer_low_risk_flight(explainer: FlightExplainer) -> None:
    """Test that a morning departure under clear VMC weather produces low delay risk."""
    low_risk_flight = {
        "op_unique_carrier": "DL",
        "origin": "ATL",
        "dest": "MCO",
        "dep_hour": 6,
        "day_of_week": 2,
        "month": 4,
        "distance": 404.0,
        "crs_elapsed_time_clean": 90.0,
        "weather_temp_f": 68.0,
        "weather_dew_point_f": 45.0,
        "weather_wind_speed_kt": 4.0,
        "weather_wind_gust_kt": 0.0,
        "weather_visibility_sm": 10.0,
        "weather_altimeter_in_hg": 30.05,
        "weather_has_snow": 0,
        "weather_has_precip": 0,
        "origin_hourly_flights": 12.0,
        "carrier_hub_daily_flights": 350.0,
        "hub_daily_total_flights": 500.0,
    }

    res = explainer.explain_flight(low_risk_flight, top_n=3)

    assert res["predicted_delay_probability"] < 0.21
    assert res["alert_triggered"] is False
    assert res["risk_tier"] in ["LOW", "MODERATE"]
    assert len(res["top_mitigating_factors"]) > 0


def test_api_health(client: TestClient) -> None:
    """Verify /health endpoint returns 200 and model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert data["features_loaded"] == 41
    assert data["threshold"] == 0.21


def test_api_predict_endpoint(client: TestClient) -> None:
    """Verify /predict endpoint returns calibrated probability and SHAP explainers."""
    payload = {
        "op_unique_carrier": "AA",
        "origin": "DFW",
        "dest": "ORD",
        "dep_hour": 18,
        "day_of_week": 4,
        "month": 11,
        "distance": 802.0,
        "crs_elapsed_time_clean": 145.0,
        "weather_wind_speed_kt": 15.0,
        "weather_wind_gust_kt": 28.0,
        "origin_hourly_flights": 42.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["carrier"] == "AA"
    assert data["origin"] == "DFW"
    assert 0.0 <= data["predicted_delay_probability"] <= 1.0
    assert isinstance(data["alert_triggered"], bool)
    assert data["risk_tier"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert len(data["top_delay_reasons"]) <= 3
    assert data["inference_latency_ms"] < 250.0  # Fast inference


def test_api_batch_predict_endpoint(client: TestClient) -> None:
    """Verify /predict/batch endpoint scores multiple flights in single call."""
    payload = {
        "flights": [
            {
                "op_unique_carrier": "UA",
                "origin": "ORD",
                "dest": "SFO",
                "dep_hour": 7,
                "day_of_week": 1,
                "month": 5,
            },
            {
                "op_unique_carrier": "DL",
                "origin": "ATL",
                "dest": "JFK",
                "dep_hour": 20,
                "day_of_week": 5,
                "month": 12,
                "weather_has_thunderstorm": 1,
            },
            {
                "op_unique_carrier": "WN",
                "origin": "DEN",
                "dest": "PHX",
                "dep_hour": 14,
                "day_of_week": 3,
                "month": 8,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_flights"] == 3
    assert len(data["results"]) == 3
    assert isinstance(data["alerts_triggered"], int)
    assert data["total_latency_ms"] > 0


def test_api_serve_index_and_metadata(client: TestClient) -> None:
    """Verify that index.html and metadata endpoints serve properly."""
    # 1. Test GET /
    index_res = client.get("/")
    assert index_res.status_code == 200
    assert "FlightOps" in index_res.text
    assert "National Airspace Network" in index_res.text

    # 2. Test GET /api/hubs
    hubs_res = client.get("/api/hubs")
    assert hubs_res.status_code == 200
    hubs = hubs_res.json()
    assert len(hubs) == 50
    assert any(h["iata"] == "ORD" for h in hubs)
    assert any(h["iata"] == "ATL" for h in hubs)

    # 3. Test GET /api/carriers
    carriers_res = client.get("/api/carriers")
    assert carriers_res.status_code == 200
    carriers = carriers_res.json()
    assert len(carriers) == 14
    assert carriers["UA"] == "United Airlines"
    assert carriers["DL"] == "Delta Air Lines"


def test_explainer_distinct_routes_correct_coordinates(
    explainer: FlightExplainer,
) -> None:
    """Regression test CORR-01: Ensure distinct routes receive true airport coordinates.

    Verifies that spatial coordinates are not silently defaulted to Atlanta-LaGuardia.
    """
    # Route 1: SEA -> MIA (Pacific Northwest to South Florida)
    sea_mia = {
        "op_unique_carrier": "AA",
        "origin": "SEA",
        "dest": "MIA",
        "dep_hour": 8,
        "day_of_week": 2,
        "month": 6,
    }
    df1 = explainer.prepare_feature_row(sea_mia)

    # Route 2: DEN -> PHX (Mountain to Desert Southwest)
    den_phx = {
        "op_unique_carrier": "WN",
        "origin": "DEN",
        "dest": "PHX",
        "dep_hour": 14,
        "day_of_week": 3,
        "month": 8,
    }
    df2 = explainer.prepare_feature_row(den_phx)

    # 1. Assert coordinates match true airport locations
    assert round(df1.iloc[0]["origin_latitude"], 1) == 47.4  # SEA lat
    assert round(df1.iloc[0]["origin_longitude"], 1) == -122.3  # SEA lon
    assert round(df1.iloc[0]["dest_latitude"], 1) == 25.8  # MIA lat
    assert round(df1.iloc[0]["dest_longitude"], 1) == -80.3  # MIA lon

    assert round(df2.iloc[0]["origin_latitude"], 1) == 39.9  # DEN lat
    assert round(df2.iloc[0]["origin_longitude"], 1) == -104.7  # DEN lon
    assert round(df2.iloc[0]["dest_latitude"], 1) == 33.4  # PHX lat
    assert round(df2.iloc[0]["dest_longitude"], 1) == -112.0  # PHX lon

    # 2. Assert neither route was defaulted to ATL (33.64, -84.42) / LGA (40.77, -73.87)
    assert df1.iloc[0]["origin_latitude"] != 33.64
    assert df1.iloc[0]["dest_latitude"] != 40.77
    assert df2.iloc[0]["origin_latitude"] != 33.64
    assert df2.iloc[0]["dest_latitude"] != 40.77

    # 3. Assert distance is route-specific and realistic
    assert 2600.0 < df1.iloc[0]["distance"] < 2800.0  # SEA-MIA ~2724 mi
    assert 550.0 < df2.iloc[0]["distance"] < 650.0  # DEN-PHX ~602 mi
    assert df1.iloc[0]["distance"] != df2.iloc[0]["distance"]


def test_api_unknown_airport_error(client: TestClient) -> None:
    """Regression test CORR-01: Unknown airport without coordinates returns 422."""
    payload = {
        "op_unique_carrier": "UA",
        "origin": "ZZZ",  # Non-existent airport code
        "dest": "ORD",
        "dep_hour": 10,
        "day_of_week": 1,
        "month": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "Unknown origin airport 'ZZZ'" in data["detail"]


def test_api_explicit_coordinates_accepted(client: TestClient) -> None:
    """Regression test CORR-01: Explicit custom coordinates are accepted and scored."""
    payload = {
        "op_unique_carrier": "UA",
        "origin": "ZZZ",
        "dest": "ORD",
        "dep_hour": 10,
        "day_of_week": 1,
        "month": 5,
        "origin_latitude": 38.5,
        "origin_longitude": -90.2,
        "distance": 300.0,
        "crs_elapsed_time_clean": 75.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["predicted_delay_probability"] <= 1.0


def test_explainer_directional_route_profiles(explainer: FlightExplainer) -> None:
    """Regression test CORR-02: Backend resolves directional route profiles."""
    # Westbound: BOS -> SFO (against prevailing jetstream headwind)
    df_west = explainer.prepare_feature_row(
        {
            "op_unique_carrier": "UA",
            "origin": "BOS",
            "dest": "SFO",
            "dep_hour": 9,
            "day_of_week": 2,
            "month": 6,
        }
    )
    # Eastbound: SFO -> BOS (with prevailing jetstream tailwind)
    df_east = explainer.prepare_feature_row(
        {
            "op_unique_carrier": "UA",
            "origin": "SFO",
            "dest": "BOS",
            "dep_hour": 9,
            "day_of_week": 2,
            "month": 6,
        }
    )

    # Exact distance matches historical BTS distance
    assert df_west.iloc[0]["distance"] == 2704.0
    assert df_east.iloc[0]["distance"] == 2704.0

    # Scheduled block time reflects historical directional asymmetry (~61 min delta)
    assert df_west.iloc[0]["crs_elapsed_time_clean"] == 404.4
    assert df_east.iloc[0]["crs_elapsed_time_clean"] == 343.4
    assert (
        df_west.iloc[0]["crs_elapsed_time_clean"]
        > df_east.iloc[0]["crs_elapsed_time_clean"]
    )


def test_explainer_unprofiled_route_fallback(explainer: FlightExplainer) -> None:
    """Regression test CORR-02: Unprofiled route falls back to Haversine."""
    # Pair that exists in airports.json but not in the commercial routes profile
    df = explainer.prepare_feature_row(
        {
            "op_unique_carrier": "WN",
            "origin": "BWI",
            "dest": "ANC",  # Ted Stevens Anchorage
            "dep_hour": 14,
            "day_of_week": 3,
            "month": 7,
        }
    )
    # Haversine distance ~3360 mi
    dist = df.iloc[0]["distance"]
    assert 3300.0 < dist < 3450.0
    # Expected speed formula: 47.43 + 0.1202 * dist
    expected_duration = round(47.43 + 0.1202 * dist, 1)
    assert abs(df.iloc[0]["crs_elapsed_time_clean"] - expected_duration) < 0.2


def test_api_rejects_extra_fields(client: TestClient) -> None:
    """Regression test SCHEMA-01: Undeclared/misspelled fields trigger HTTP 422."""
    payload = {
        "op_unique_carrier": "UA",
        "origin": "ORD",
        "dest": "LGA",
        "dep_hour": 14,
        "day_of_week": 2,
        "month": 8,
        "orig_lat": 41.97,  # Misspelled field name (should be origin_latitude)
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    err_body = str(response.json())
    assert "extra_forbidden" in err_body or "Extra inputs are not permitted" in err_body


def test_api_scenario_presets_valid(client: TestClient) -> None:
    """Verify that all standard UI scenario presets succeed with valid scoring."""
    presets = [
        {"carrier": "UA", "origin": "ORD", "dest": "LGA", "hour": 19, "snow": 1},
        {"carrier": "DL", "origin": "ATL", "dest": "MCO", "hour": 7, "snow": 0},
        {"carrier": "WN", "origin": "DEN", "dest": "SFO", "hour": 17, "snow": 1},
        {"carrier": "AA", "origin": "DFW", "dest": "LAX", "hour": 18, "snow": 0},
    ]
    for p in presets:
        payload = {
            "op_unique_carrier": p["carrier"],
            "origin": p["origin"],
            "dest": p["dest"],
            "dep_hour": p["hour"],
            "day_of_week": 4,
            "month": 11,
            "weather_has_snow": p["snow"],
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data["predicted_delay_probability"] <= 1.0
        assert data["risk_tier"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]
        assert len(data["top_delay_reasons"]) > 0

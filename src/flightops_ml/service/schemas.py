"""Pydantic schemas for FlightOps ML REST service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlightPredictionRequest(BaseModel):
    """Input payload for scoring a single flight 2 hours prior to departure."""

    model_config = ConfigDict(extra="forbid")

    op_unique_carrier: str = Field(
        ..., description="Operating carrier code (e.g., UA, AA, DL, WN)"
    )
    origin: str = Field(
        ..., description="Origin airport IATA hub code (e.g., ORD, ATL, DFW)"
    )
    dest: str = Field(..., description="Destination airport IATA code")
    dep_hour: int = Field(
        ..., ge=0, le=23, description="Scheduled departure hour (0-23 in local time)"
    )
    day_of_week: int = Field(
        ..., ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)"
    )
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    distance: float | None = Field(
        None,
        ge=20.0,
        le=6000.0,
        description="Scheduled flight distance (miles, auto-resolved if omitted)",
    )
    crs_elapsed_time_clean: float | None = Field(
        None,
        ge=20.0,
        le=800.0,
        description="Scheduled flight duration (minutes, auto-resolved if omitted)",
    )
    origin_latitude: float | None = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Origin airport latitude (auto-resolved if omitted)",
    )
    origin_longitude: float | None = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Origin airport longitude (auto-resolved if omitted)",
    )
    dest_latitude: float | None = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Destination airport latitude (auto-resolved if omitted)",
    )
    dest_longitude: float | None = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Destination airport longitude (auto-resolved if omitted)",
    )

    # Optional Terminal ASOS Weather at T-2h (Standard VMC if missing)
    weather_temp_f: float = Field(65.0, description="Surface temperature in Fahrenheit")
    weather_dew_point_f: float = Field(
        50.0, description="Surface dew point in Fahrenheit"
    )
    weather_humidity_pct: float = Field(
        60.0, description="Relative humidity percentage"
    )
    weather_wind_dir_deg: float = Field(
        0.0, description="Wind direction in degrees (0-360)"
    )
    weather_wind_speed_kt: float = Field(
        8.0, description="Sustained surface wind speed in knots"
    )
    weather_wind_gust_kt: float = Field(0.0, description="Peak wind gust in knots")
    weather_visibility_sm: float = Field(
        10.0, description="Surface visibility in statute miles"
    )
    weather_altimeter_in_hg: float = Field(
        29.92, description="Altimeter barometric pressure in inHg"
    )
    weather_precip_1h_in: float = Field(
        0.0, description="Liquid equivalent precipitation in past 1h (inches)"
    )
    weather_has_precip: int = Field(
        0, description="1 if precipitation is present, 0 otherwise"
    )
    weather_has_thunderstorm: int = Field(
        0, description="1 if active thunderstorm at hub, 0 otherwise"
    )
    weather_has_snow: int = Field(
        0, description="1 if active snowfall or freezing precipitation, 0 otherwise"
    )
    weather_has_fog_haze: int = Field(
        0, description="1 if active fog or dense haze, 0 otherwise"
    )

    # Optional Airport Traffic Congestion at T-2h
    origin_hourly_flights: float = Field(
        25.0, ge=1.0, description="Scheduled departures from origin hub in this hour"
    )
    carrier_hub_daily_flights: float = Field(
        100.0, ge=1.0, description="Carrier total daily flights at this origin"
    )
    hub_daily_total_flights: float = Field(
        300.0, ge=1.0, description="All carriers daily departures at this origin"
    )


class DelayReason(BaseModel):
    """Structured explanation for a top feature driver."""

    feature: str
    display_name: str
    feature_value: Any
    shap_impact: float
    direction: str
    description: str


class FlightPredictionResponse(BaseModel):
    """Prediction output with probability, alert status, and SHAP explainers."""

    carrier: str
    origin: str
    dest: str
    dep_hour: int
    predicted_delay_probability: float
    alert_triggered: bool
    risk_tier: str
    operational_threshold: float
    top_delay_reasons: list[DelayReason]
    top_mitigating_factors: list[DelayReason]
    inference_latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Batch input payload."""

    model_config = ConfigDict(extra="forbid")
    flights: list[FlightPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    total_flights: int
    alerts_triggered: int
    results: list[FlightPredictionResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Service health and model metadata."""

    status: str
    model_version: str
    features_loaded: int
    threshold: float

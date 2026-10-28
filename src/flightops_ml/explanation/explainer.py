"""TreeSHAP flight delay explainer module.

Translates LightGBM TreeSHAP values into human-readable delay drivers
and mitigating factors for flight operations dispatchers.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
import shap

LOGGER = logging.getLogger("flightops_ml.explanation")


class FlightExplainer:
    """TreeSHAP explainer for single and batch flight delay risk predictions."""

    def __init__(
        self,
        model_path: str | Path = "models/lgb_champion.txt",
        metadata_path: str | Path = "models/model_metadata.json",
        airports_path: str | Path | None = None,
        routes_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        LOGGER.info("Loading LightGBM production booster from: %s", self.model_path)
        self.booster = lgb.Booster(model_file=str(self.model_path))

        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.feature_cols: list[str] = self.metadata["feature_cols"]
        self.cat_cols: list[str] = self.metadata["cat_cols"]
        self.global_base_rate: float = self.metadata["global_train_base_rate"]
        self.threshold: float = self.metadata.get("operational_threshold_tau", 0.21)

        self.prior_carrier_hour: dict[str, float] = self.metadata["prior_carrier_hour"]
        self.prior_origin_hour: dict[str, float] = self.metadata["prior_origin_hour"]
        self.prior_carrier_origin: dict[str, float] = self.metadata[
            "prior_carrier_origin"
        ]

        # Authoritative airport reference and historical route profiles
        static_dir = Path(__file__).resolve().parent.parent / "service" / "static"
        airports_file = (
            Path(airports_path) if airports_path else static_dir / "airports.json"
        )
        routes_file = (
            Path(routes_path) if routes_path else static_dir / "routes_reference.json"
        )

        self.airports: dict[str, dict[str, Any]] = {}
        if airports_file.exists():
            with open(airports_file, encoding="utf-8") as f:
                self.airports = json.load(f)

        self.routes: dict[str, dict[str, float]] = {}
        if routes_file.exists():
            with open(routes_file, encoding="utf-8") as f:
                self.routes = json.load(f)

        LOGGER.info(
            "Loaded %d reference airports and %d route profiles.",
            len(self.airports),
            len(self.routes),
        )

        LOGGER.info("Initializing TreeSHAP Explainer...")
        self.explainer = shap.TreeExplainer(self.booster)
        LOGGER.info("FlightExplainer initialized successfully.")

    @staticmethod
    def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute great-circle distance in statute miles via Haversine formula."""
        r_earth = 3958.8
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(r_earth * c, 1)

    def prepare_feature_row(self, data: dict[str, Any]) -> pd.DataFrame:
        """Enrich raw flight input with features, coordinates, and priors."""
        carrier = str(data.get("op_unique_carrier", "WN")).upper().strip()
        origin = str(data.get("origin", "ATL")).upper().strip()
        dest = str(data.get("dest", "LGA")).upper().strip()
        dep_hour = int(data.get("dep_hour", 12))
        day_of_week = int(data.get("day_of_week", 1))

        # 1. Resolve airport coordinates
        orig_lat = data.get("origin_latitude")
        orig_lon = data.get("origin_longitude")
        dest_lat = data.get("dest_latitude")
        dest_lon = data.get("dest_longitude")

        if orig_lat is None or orig_lon is None:
            if origin in self.airports:
                orig_lat = self.airports[origin]["lat"]
                orig_lon = self.airports[origin]["lon"]
            else:
                raise ValueError(
                    f"Unknown origin airport '{origin}'. Coordinates must be supplied."
                )

        if dest_lat is None or dest_lon is None:
            if dest in self.airports:
                dest_lat = self.airports[dest]["lat"]
                dest_lon = self.airports[dest]["lon"]
            else:
                raise ValueError(
                    f"Unknown destination airport '{dest}'. "
                    "Coordinates must be supplied."
                )

        orig_lat = float(orig_lat)
        orig_lon = float(orig_lon)
        dest_lat = float(dest_lat)
        dest_lon = float(dest_lon)

        # 2. Resolve distance and scheduled elapsed duration
        distance = data.get("distance")
        elapsed = data.get("crs_elapsed_time_clean")

        route_key = f"{origin}_{dest}"
        if distance is None:
            if route_key in self.routes:
                distance = self.routes[route_key]["distance"]
            else:
                distance = self.haversine_miles(orig_lat, orig_lon, dest_lat, dest_lon)
        distance = float(distance)

        if elapsed is None:
            if route_key in self.routes:
                elapsed = self.routes[route_key]["crs_elapsed_time_clean"]
            else:
                elapsed = round(47.43 + 0.1202 * distance, 1)
        elapsed = float(elapsed)

        # Cyclic features
        dep_time_rad = 2 * math.pi * dep_hour / 24.0
        dow_rad = 2 * math.pi * day_of_week / 7.0
        month = int(data.get("month", 6))
        month_rad = 2 * math.pi * month / 12.0

        sin_dep_time = math.sin(dep_time_rad)
        cos_dep_time = math.cos(dep_time_rad)
        sin_day_of_week = math.sin(dow_rad)
        cos_day_of_week = math.cos(dow_rad)
        sin_month = math.sin(month_rad)
        cos_month = math.cos(month_rad)

        # Weather features with defaults
        temp_f = float(data.get("weather_temp_f", 65.0))
        dew_point_f = float(data.get("weather_dew_point_f", 50.0))
        humidity_pct = float(data.get("weather_humidity_pct", 60.0))
        wind_dir = float(data.get("weather_wind_dir_deg", 0.0))
        wind_speed = float(data.get("weather_wind_speed_kt", 8.0))
        wind_gust = float(data.get("weather_wind_gust_kt", 0.0))
        visibility = float(data.get("weather_visibility_sm", 10.0))
        altimeter = float(data.get("weather_altimeter_in_hg", 29.92))
        precip_1h = float(data.get("weather_precip_1h_in", 0.0))
        has_precip = float(data.get("weather_has_precip", 1 if precip_1h > 0 else 0))
        has_ts = float(data.get("weather_has_thunderstorm", 0))
        has_snow = float(data.get("weather_has_snow", 0))
        has_fog = float(data.get("weather_has_fog_haze", 0))

        # Traffic & Hub Congestion
        origin_hourly = float(data.get("origin_hourly_flights", 25.0))
        carrier_hub_daily = float(data.get("carrier_hub_daily_flights", 100.0))
        hub_daily_total = float(data.get("hub_daily_total_flights", 300.0))
        carrier_hub_share = carrier_hub_daily / max(1.0, hub_daily_total)

        # Derived physics
        wind_gust_spread = max(0.0, wind_gust - wind_speed)
        temp_dewpoint_spread = temp_f - dew_point_f
        pressure_anomaly = altimeter - 29.92
        severe_weather_composite = (
            has_ts * 3.0
            + has_snow * 2.0
            + has_fog * 1.5
            + (2.0 if wind_gust >= 25.0 else 0.0)
        )

        # Bayesian Priors
        p_c_h = self.prior_carrier_hour.get(
            f"{carrier}_{dep_hour}", self.global_base_rate
        )
        p_o_h = self.prior_origin_hour.get(
            f"{origin}_{dep_hour}", self.global_base_rate
        )
        p_c_o = self.prior_carrier_origin.get(
            f"{carrier}_{origin}", self.global_base_rate
        )

        row_dict = {
            "op_unique_carrier": carrier,
            "origin": origin,
            "dep_hour": dep_hour,
            "day_of_week": day_of_week,
            "is_weekend": float(
                data.get("is_weekend", 1 if day_of_week in [6, 7] else 0)
            ),
            "sin_dep_time": sin_dep_time,
            "cos_dep_time": cos_dep_time,
            "sin_day_of_week": sin_day_of_week,
            "cos_day_of_week": cos_day_of_week,
            "sin_month": sin_month,
            "cos_month": cos_month,
            "distance": distance,
            "crs_elapsed_time_clean": elapsed,
            "origin_latitude": orig_lat,
            "origin_longitude": orig_lon,
            "dest_latitude": dest_lat,
            "dest_longitude": dest_lon,
            "weather_temp_f": temp_f,
            "weather_dew_point_f": dew_point_f,
            "weather_humidity_pct": humidity_pct,
            "weather_wind_dir_deg": wind_dir,
            "weather_wind_speed_kt": wind_speed,
            "weather_wind_gust_kt": wind_gust,
            "weather_visibility_sm": visibility,
            "weather_altimeter_in_hg": altimeter,
            "weather_precip_1h_in": precip_1h,
            "weather_has_precip": has_precip,
            "weather_has_thunderstorm": has_ts,
            "weather_has_snow": has_snow,
            "weather_has_fog_haze": has_fog,
            "origin_hourly_flights": origin_hourly,
            "carrier_hub_daily_flights": carrier_hub_daily,
            "hub_daily_total_flights": hub_daily_total,
            "carrier_hub_share": carrier_hub_share,
            "wind_gust_spread": wind_gust_spread,
            "temp_dewpoint_spread": temp_dewpoint_spread,
            "pressure_anomaly": pressure_anomaly,
            "severe_weather_composite": severe_weather_composite,
            "prior_carrier_hour_risk": p_c_h,
            "prior_origin_hour_risk": p_o_h,
            "prior_carrier_origin_risk": p_c_o,
        }

        df = pd.DataFrame([row_dict])[self.feature_cols]
        for col in self.cat_cols:
            df[col] = df[col].astype("category")
        return df

    def explain_flight(self, data: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
        """Score a flight and return delay probability with top SHAP delay reasons."""
        df = self.prepare_feature_row(data)

        prob = float(self.booster.predict(df)[0])
        shap_explanation = self.explainer(df)
        shap_vals = shap_explanation.values[0]

        # Determine operational risk tier
        if prob < 0.15:
            risk_tier = "LOW"
        elif prob < self.threshold:
            risk_tier = "MODERATE"
        elif prob < 0.35:
            risk_tier = "HIGH"
        else:
            risk_tier = "SEVERE"

        # Pair features with values and SHAP scores
        contributions = []
        for feat_name, shap_val in zip(self.feature_cols, shap_vals, strict=False):
            feat_val = df.iloc[0][feat_name]
            contributions.append(
                {
                    "feature": feat_name,
                    "value": feat_val,
                    "shap_value": float(shap_val),
                }
            )

        # Separate risk-increasing (positive SHAP) vs risk-decreasing (negative SHAP)
        risk_drivers = [c for c in contributions if c["shap_value"] > 0]
        risk_drivers.sort(key=lambda x: x["shap_value"], reverse=True)

        mitigating_drivers = [c for c in contributions if c["shap_value"] < 0]
        mitigating_drivers.sort(key=lambda x: x["shap_value"])

        # Translate top drivers into clean operational explanations
        top_delay_reasons = [
            self._describe_feature(
                c["feature"], c["value"], c["shap_value"], is_delay_driver=True
            )
            for c in risk_drivers[:top_n]
        ]

        top_mitigators = [
            self._describe_feature(
                c["feature"], c["value"], c["shap_value"], is_delay_driver=False
            )
            for c in mitigating_drivers[:top_n]
        ]

        return {
            "flight_info": {
                "carrier": str(data.get("op_unique_carrier", "Unknown")),
                "origin": str(data.get("origin", "Unknown")),
                "dest": str(data.get("dest", "Unknown")),
                "dep_hour": int(data.get("dep_hour", 12)),
            },
            "predicted_delay_probability": round(prob, 4),
            "alert_triggered": bool(prob >= self.threshold),
            "risk_tier": risk_tier,
            "operational_threshold": self.threshold,
            "top_delay_reasons": top_delay_reasons,
            "top_mitigating_factors": top_mitigators,
        }

    @staticmethod
    def _describe_feature(
        name: str, val: Any, shap_val: float, is_delay_driver: bool
    ) -> dict[str, Any]:
        """Generate human-readable operational description for a feature."""
        direction = "RISK_INCREASING" if is_delay_driver else "RISK_DECREASING"
        impact_score = round(abs(shap_val), 4)

        if name == "origin_hourly_flights":
            display_name = "Runway Congestion"
            description = f"High queue with {int(float(val))} flights in origin hour."
        elif name == "wind_gust_spread":
            display_name = "Wind Gust Shear"
            description = f"Gusts exceed sustained wind by {float(val):.1f} kt."
        elif name == "weather_wind_gust_kt":
            display_name = "Surface Wind Gusts"
            description = f"Peak gusts of {float(val):.0f} kt recorded at origin hub."
        elif name == "weather_wind_speed_kt":
            display_name = "High Surface Winds"
            description = f"Sustained wind speeds of {float(val):.0f} kt at runway."
        elif name == "weather_visibility_sm":
            display_name = "Low Runway Visibility"
            description = f"Restricted visibility of only {float(val):.1f} miles."
        elif name == "weather_has_thunderstorm":
            display_name = "Terminal Thunderstorm"
            description = "Active thunderstorm activity reported at origin hub."
        elif name == "weather_has_snow":
            display_name = "Winter Precipitation"
            description = "Active snowfall requiring runway and aircraft de-icing."
        elif name == "weather_has_fog_haze":
            display_name = "Low Fog & Ceiling"
            description = "Reduced visual meteorological conditions due to fog/haze."
        elif name == "weather_precip_1h_in":
            display_name = "Surface Rain Accumulation"
            description = f"Heavy rain measuring {float(val):.2f} inches in past hour."
        elif name == "dep_hour":
            display_name = "Departure Time Peak"
            description = (
                f"Scheduled at {int(float(val)):02d}:00 during peak delay window."
            )
        elif name == "prior_origin_hour_risk":
            display_name = "Origin Hub Delay Baseline"
            description = f"Hub averages {float(val) * 100:.1f}% delay at this hour."
        elif name == "prior_carrier_hour_risk":
            display_name = "Carrier Delay Baseline"
            description = (
                f"Carrier averages {float(val) * 100:.1f}% delay at this hour."
            )
        elif name == "prior_carrier_origin_risk":
            display_name = "Carrier-Hub Turnaround History"
            description = f"Carrier averages {float(val) * 100:.1f}% delays at origin."
        elif name == "carrier_hub_share":
            display_name = "Carrier Hub Concentration"
            description = (
                f"Carrier operates {float(val) * 100:.1f}% of daily flights at hub."
            )
        elif name == "severe_weather_composite":
            display_name = "Compound Weather Hazard"
            description = f"Multi-hazard atmospheric index score of {float(val):.1f}."
        elif name == "temp_dewpoint_spread":
            display_name = "Temperature-Dewpoint Saturation"
            description = f"Narrow {float(val):.1f}°F spread indicating fog risk."
        elif name == "pressure_anomaly":
            display_name = "Cyclonic Low Pressure"
            description = f"Pressure anomaly of {float(val):+.2f} inHg below standard."
        elif name == "distance":
            display_name = "Long Haul Route"
            description = f"Flight distance of {float(val):.0f} miles."
        elif name == "crs_elapsed_time_clean":
            display_name = "Scheduled Flight Time"
            description = f"Scheduled block time of {float(val):.0f} minutes."
        elif name == "origin":
            display_name = "Origin Hub Location"
            description = f"Operations departing from {val} hub."
        elif name == "op_unique_carrier":
            display_name = "Operating Carrier Network"
            description = f"Flight operated by carrier {val}."
        elif name in (
            "origin_latitude",
            "origin_longitude",
            "dest_latitude",
            "dest_longitude",
        ):
            display_name = "Route Geography"
            description = f"Spatial coordinate of route corridor ({float(val):.2f}°)."
        else:
            display_name = name.replace("_", " ").title()
            description = f"{display_name} contributes {shap_val:+.3f} to delay risk."

        # Mitigating drivers
        if not is_delay_driver:
            try:
                numeric_val = float(val)
                if name == "dep_hour" and numeric_val <= 9:
                    display_name = "Early Morning Departure"
                    description = (
                        f"Scheduled at {int(numeric_val):02d}:00 with fresh aircraft."
                    )
                elif name == "origin_hourly_flights" and numeric_val < 15:
                    display_name = "Clear Runway Capacity"
                    description = (
                        f"Light traffic with only {int(numeric_val)} hourly flights."
                    )
                elif "weather" in name and (numeric_val == 0 or numeric_val >= 10):
                    display_name = "Favorable VMC Weather"
                    description = "Clear skies, good visibility, and calm winds."
            except (ValueError, TypeError):
                pass

        return {
            "feature": name,
            "display_name": display_name,
            "feature_value": val,
            "shap_impact": impact_score,
            "direction": direction,
            "description": description,
        }

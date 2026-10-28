"""FastAPI flight departure delay prediction service.

Delivers real-time inference and TreeSHAP explainability (Top 3 Delay Reasons)
for airline dispatchers and operations controllers at the strict T - 2h cutoff.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flightops_ml.explanation.explainer import FlightExplainer
from flightops_ml.service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FlightPredictionRequest,
    FlightPredictionResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
LOGGER = logging.getLogger("flightops_ml.service")

# Global explainer instance
EXPLAINER: FlightExplainer | None = None
STATIC_DIR = Path(__file__).parent / "static"
FIGURES_DIR = Path("docs/figures")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load model artifacts and initialize TreeSHAP explainer on startup."""
    global EXPLAINER
    LOGGER.info("Initializing FlightOps ML inference service...")
    try:
        EXPLAINER = FlightExplainer(
            model_path="models/lgb_champion.txt",
            metadata_path="models/model_metadata.json",
        )
        LOGGER.info("Model and TreeSHAP engine loaded into memory successfully.")
    except Exception as exc:
        LOGGER.warning("Could not load production model on startup: %s", exc)
        EXPLAINER = None
    yield
    LOGGER.info("Shutting down FlightOps ML inference service...")


app = FastAPI(
    title="FlightOps ML — Flight Delay Risk & Explainability API",
    description=(
        "REST service predicting flight departure delays (>=15 min) at 2 hours "
        "before scheduled departure, returning calibrated risk probabilities and "
        "top TreeSHAP operational delay drivers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets and figures
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the interactive FlightOps ML control room and dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard index.html not found.",
        )
    return FileResponse(index_path)


@app.get("/api/hubs", tags=["Metadata"])
async def get_hubs() -> list[dict[str, Any]]:
    """Return list of Top 50 Hubs with coordinates and baseline delay rates."""
    hubs_path = STATIC_DIR / "hubs.json"
    if not hubs_path.exists():
        return []
    with open(hubs_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/carriers", tags=["Metadata"])
async def get_carriers() -> dict[str, str]:
    """Return dictionary of 14 supported carriers with airline names."""
    return {
        "AA": "American Airlines",
        "AS": "Alaska Airlines",
        "B6": "JetBlue Airways",
        "DL": "Delta Air Lines",
        "F9": "Frontier Airlines",
        "G4": "Allegiant Air",
        "HA": "Hawaiian Airlines",
        "MQ": "Envoy Air",
        "NK": "Spirit Airlines",
        "OO": "SkyWest Airlines",
        "UA": "United Airlines",
        "WN": "Southwest Airlines",
        "YV": "Mesa Airlines",
        "YX": "Republic Airways",
    }


def get_explainer() -> FlightExplainer:
    """Dependency helper to get the initialized explainer."""
    global EXPLAINER
    if EXPLAINER is None:
        try:
            EXPLAINER = FlightExplainer(
                model_path="models/lgb_champion.txt",
                metadata_path="models/model_metadata.json",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model engine is not loaded: {exc}",
            ) from exc
    return EXPLAINER


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Service health check and model version."""
    explainer = get_explainer()
    return HealthResponse(
        status="healthy",
        model_version=explainer.metadata.get("model_version", "1.0.0"),
        features_loaded=len(explainer.feature_cols),
        threshold=explainer.threshold,
    )


@app.post("/predict", response_model=FlightPredictionResponse, tags=["Inference"])
async def predict_flight(request: FlightPredictionRequest) -> FlightPredictionResponse:
    """Score a flight at T - 2h and return delay probability with top 3 explainers.

    Returns:
    - predicted_delay_probability: Calibrated delay probability (0.0 to 1.0).
    - alert_triggered: True if probability >= operational threshold (tau* = 0.21).
    - risk_tier: LOW, MODERATE, HIGH, or SEVERE.
    - top_delay_reasons: Top 3 factors driving delay probability UP.
    - top_mitigating_factors: Top factors pulling delay probability DOWN.
    """
    t0 = time.perf_counter()
    explainer = get_explainer()

    try:
        raw_dict = request.model_dump()
        result = explainer.explain_flight(raw_dict, top_n=3)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return FlightPredictionResponse(
            carrier=result["flight_info"]["carrier"],
            origin=result["flight_info"]["origin"],
            dest=result["flight_info"]["dest"],
            dep_hour=result["flight_info"]["dep_hour"],
            predicted_delay_probability=result["predicted_delay_probability"],
            alert_triggered=result["alert_triggered"],
            risk_tier=result["risk_tier"],
            operational_threshold=result["operational_threshold"],
            top_delay_reasons=result["top_delay_reasons"],
            top_mitigating_factors=result["top_mitigating_factors"],
            inference_latency_ms=round(latency_ms, 2),
        )
    except ValueError as exc:
        LOGGER.warning("Inference validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference input validation error: {exc}",
        ) from exc
    except Exception as exc:
        LOGGER.error("Inference failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch scoring for multiple flights across a departure bank."""
    t0 = time.perf_counter()
    explainer = get_explainer()

    results: list[FlightPredictionResponse] = []
    alerts_count = 0

    try:
        for flight in request.flights:
            t_sub = time.perf_counter()
            raw_dict = flight.model_dump()
            res = explainer.explain_flight(raw_dict, top_n=3)
            sub_lat = (time.perf_counter() - t_sub) * 1000.0

            if res["alert_triggered"]:
                alerts_count += 1

            results.append(
                FlightPredictionResponse(
                    carrier=res["flight_info"]["carrier"],
                    origin=res["flight_info"]["origin"],
                    dest=res["flight_info"]["dest"],
                    dep_hour=res["flight_info"]["dep_hour"],
                    predicted_delay_probability=res["predicted_delay_probability"],
                    alert_triggered=res["alert_triggered"],
                    risk_tier=res["risk_tier"],
                    operational_threshold=res["operational_threshold"],
                    top_delay_reasons=res["top_delay_reasons"],
                    top_mitigating_factors=res["top_mitigating_factors"],
                    inference_latency_ms=round(sub_lat, 2),
                )
            )
    except ValueError as exc:
        LOGGER.warning("Batch inference validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch input validation error: {exc}",
        ) from exc
    except Exception as exc:
        LOGGER.error("Batch inference failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {exc}",
        ) from exc

    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    return BatchPredictionResponse(
        total_flights=len(results),
        alerts_triggered=alerts_count,
        results=results,
        total_latency_ms=round(total_latency_ms, 2),
    )

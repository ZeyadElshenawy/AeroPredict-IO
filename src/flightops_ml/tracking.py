"""FlightOps ML — MLflow Experiment Tracking & Model Registry Integration.

Provides standardized tracking URI resolution, experiment registration, run logging,
and model registry management for AeroPredict IO™ models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger("flightops_ml.tracking")

DEFAULT_TRACKING_DB = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_DIR = "mlruns"
DEFAULT_EXPERIMENT_NAME = "flightops-delay-prediction"
DEFAULT_MODEL_REGISTRY_NAME = "flightops-departure-delay"


def get_tracking_uri() -> str:
    """Return the configured MLflow tracking URI."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return DEFAULT_TRACKING_DB


def setup_tracking(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> tuple[str, str]:
    """Configure MLflow tracking URI and ensure the target experiment exists."""
    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)

    client = MlflowClient(tracking_uri=uri)
    experiment: Experiment | None = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=DEFAULT_ARTIFACT_DIR,
        )
        LOGGER.info(
            "Created MLflow Experiment '%s' (ID: %s)", experiment_name, experiment_id
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    LOGGER.info(
        "MLflow configured: URI=%s, Experiment='%s' (ID: %s)",
        uri,
        experiment_name,
        experiment_id,
    )
    return uri, experiment_id


def log_experiment_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: list[str | Path] | None = None,
    model: Any | None = None,
    model_flavor: str = "lightgbm",
    artifact_path: str = "model",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> Run:
    """Execute and log a complete run to MLflow."""
    setup_tracking(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("platform", "AeroPredict-IO")

        if params:
            clean_params = {
                k: str(v) if isinstance(v, (list, dict, tuple)) else v
                for k, v in params.items()
            }
            mlflow.log_params(clean_params)

        if metrics:
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if v is not None}
            )

        if artifacts:
            for art in artifacts:
                art_path = Path(art)
                if art_path.exists():
                    mlflow.log_artifact(str(art_path))
                    LOGGER.info(
                        "Logged artifact to run %s: %s",
                        run.info.run_id,
                        art_path.name,
                    )
                else:
                    LOGGER.warning("Artifact path not found, skipping: %s", art_path)

        if model is not None:
            if model_flavor == "lightgbm":
                mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
            elif model_flavor == "catboost":
                mlflow.catboost.log_model(model, artifact_path=artifact_path)
            elif model_flavor == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            else:
                LOGGER.warning(
                    "Unknown model flavor '%s'; skipping model save", model_flavor
                )

        LOGGER.info(
            "Logged MLflow run '%s' (ID: %s, metrics: %d)",
            run_name,
            run.info.run_id,
            len(metrics) if metrics else 0,
        )
        return run


def register_model_champion(
    run_id: str,
    model_name: str = DEFAULT_MODEL_REGISTRY_NAME,
    model_subpath: str = "model",
    alias: str = "champion",
    description: str | None = None,
) -> Any:
    """Register a run's model into the MLflow Model Registry and assign an alias."""
    uri = get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)

    model_source = f"runs:/{run_id}/{model_subpath}"
    LOGGER.info("Registering model '%s' from source: %s", model_name, model_source)

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            name=model_name,
            description="AeroPredict IO™ Departure Delay Risk Model Registry",
        )

    version = client.create_model_version(
        name=model_name,
        source=model_source,
        run_id=run_id,
        description=description or "Production Champion Model (LightGBM)",
    )

    client.set_registered_model_alias(
        name=model_name, alias=alias, version=version.version
    )
    if alias == "champion":
        client.set_registered_model_alias(
            name=model_name, alias="production", version=version.version
        )

    LOGGER.info(
        "Model '%s' v%s registered with alias '%s'",
        model_name,
        version.version,
        alias,
    )
    return version

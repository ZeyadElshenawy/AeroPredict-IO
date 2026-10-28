"""Tests for flightops_ml.tracking (MLflow experiment tracking and registry)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from flightops_ml.tracking import (
    get_tracking_uri,
    log_experiment_run,
    register_model_champion,
    setup_tracking,
)


def test_get_tracking_uri_default(monkeypatch):
    """Test get_tracking_uri returns env var when set, or default sqlite DB."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = get_tracking_uri()
    assert "sqlite:///" in uri

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test_custom.db")
    assert get_tracking_uri() == "sqlite:///test_custom.db"


def test_setup_tracking_and_log_run(tmp_path: Path):
    """Test setup_tracking and logging a run with params, metrics, and tags."""
    test_db = tmp_path / "test_mlflow.db"
    test_uri = f"sqlite:///{test_db.as_posix()}"

    uri, exp_id = setup_tracking(
        experiment_name="test-flightops-experiment",
        tracking_uri=test_uri,
    )
    assert uri == test_uri
    assert exp_id is not None

    dummy_art = tmp_path / "dummy_plot.txt"
    dummy_art.write_text("dummy artifact content", encoding="utf-8")

    run = log_experiment_run(
        run_name="Unit-Test-Run",
        params={"learning_rate": 0.05, "num_leaves": 31},
        metrics={"pr_auc": 0.4195, "roc_auc": 0.6949},
        tags={"stage": "test"},
        artifacts=[dummy_art],
        experiment_name="test-flightops-experiment",
    )

    client = MlflowClient(tracking_uri=test_uri)
    fetched_run = client.get_run(run.info.run_id)

    assert fetched_run.info.run_id is not None
    assert fetched_run.data.params["learning_rate"] == "0.05"
    assert pytest.approx(fetched_run.data.metrics["pr_auc"], 0.001) == 0.4195


def test_model_logging_and_registry(tmp_path: Path):
    """Test logging a real LightGBM booster and registering it in the registry."""
    test_db = tmp_path / "test_registry.db"
    test_uri = f"sqlite:///{test_db.as_posix()}"

    setup_tracking(
        experiment_name="test-registry-experiment",
        tracking_uri=test_uri,
    )

    # Train trivial booster
    rng = np.random.default_rng(42)
    X = rng.random((50, 4))
    y = (X[:, 0] > 0.5).astype(int)
    ds = lgb.Dataset(X, label=y)
    booster = lgb.train({"objective": "binary", "verbose": -1}, ds, num_boost_round=3)

    run = log_experiment_run(
        run_name="Trivial-Booster-Run",
        params={"trees": 3},
        metrics={"accuracy": 0.9},
        model=booster,
        model_flavor="lightgbm",
        experiment_name="test-registry-experiment",
    )

    # Register champion
    reg = register_model_champion(
        run_id=run.info.run_id,
        model_name="test-delay-model",
        alias="champion",
    )

    assert reg.name == "test-delay-model"
    assert str(reg.version) == "1"

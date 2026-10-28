"""Train and export the production LightGBM model and SHAP explainer metadata.

Saves:
- models/lgb_champion.txt (LightGBM booster)
- models/model_metadata.json (Bayesian priors, feature columns, thresholds)
- docs/figures/15_shap_summary.png (Global SHAP feature impact beeswarm)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from flightops_ml.ingestion.loader import database_connection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
LOGGER = logging.getLogger("train_production_model")

MODELS_DIR = Path("models")
FIGURES_DIR = Path("docs/figures")


def main() -> None:
    t_start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Connecting to PostgreSQL and loading Train and Validation data...")
    query = """
        SELECT
            w.dataset_split,
            w.dep_del15,
            w.flight_date,
            w.op_unique_carrier,
            w.origin,
            w.dep_hour,
            w.day_of_week,
            w.is_weekend,
            w.sin_dep_time,
            w.cos_dep_time,
            w.sin_day_of_week,
            w.cos_day_of_week,
            w.sin_month,
            w.cos_month,
            w.distance,
            w.crs_elapsed_time_clean,
            w.origin_latitude,
            w.origin_longitude,
            w.dest_latitude,
            w.dest_longitude,
            COALESCE(w.weather_temp_f, 65.0) as weather_temp_f,
            COALESCE(w.weather_dew_point_f, 50.0) as weather_dew_point_f,
            COALESCE(w.weather_humidity_pct, 60.0) as weather_humidity_pct,
            COALESCE(w.weather_wind_dir_deg, 0.0) as weather_wind_dir_deg,
            COALESCE(w.weather_wind_speed_kt, 8.0) as weather_wind_speed_kt,
            COALESCE(w.weather_wind_gust_kt, 0.0) as weather_wind_gust_kt,
            COALESCE(w.weather_visibility_sm, 10.0) as weather_visibility_sm,
            COALESCE(w.weather_altimeter_in_hg, 29.92) as weather_altimeter_in_hg,
            COALESCE(w.weather_precip_1h_in, 0.0) as weather_precip_1h_in,
            COALESCE(w.weather_has_precip, 0) as weather_has_precip,
            COALESCE(w.weather_has_thunderstorm, 0) as weather_has_thunderstorm,
            COALESCE(w.weather_has_snow, 0) as weather_has_snow,
            COALESCE(w.weather_has_fog_haze, 0) as weather_has_fog_haze,
            COUNT(*) OVER (PARTITION BY w.origin, w.flight_date, w.dep_hour) as origin_hourly_flights,
            COUNT(*) OVER (PARTITION BY w.op_unique_carrier, w.origin, w.flight_date) as carrier_hub_daily_flights,
            COUNT(*) OVER (PARTITION BY w.origin, w.flight_date) as hub_daily_total_flights
        FROM public.vw_gold_top50_weather w
        WHERE w.dataset_split IN ('train', 'validation');
    """

    with database_connection() as conn:
        t0 = time.time()
        df = pd.read_sql(query, conn)
        LOGGER.info("Loaded %s rows in %.2fs", f"{len(df):,}", time.time() - t0)

    # In-memory domain features
    df["carrier_hub_share"] = df["carrier_hub_daily_flights"] / df[
        "hub_daily_total_flights"
    ].replace(0, 1)
    df["wind_gust_spread"] = np.maximum(
        0.0, df["weather_wind_gust_kt"] - df["weather_wind_speed_kt"]
    )
    df["temp_dewpoint_spread"] = df["weather_temp_f"] - df["weather_dew_point_f"]
    df["pressure_anomaly"] = df["weather_altimeter_in_hg"] - 29.92
    df["severe_weather_composite"] = (
        df["weather_has_thunderstorm"] * 3.0
        + df["weather_has_snow"] * 2.0
        + df["weather_has_fog_haze"] * 1.5
        + (df["weather_wind_gust_kt"] >= 25.0).astype(float) * 2.0
    )

    train_mask = df["dataset_split"] == "train"
    val_mask = df["dataset_split"] == "validation"

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    y_train = train_df["dep_del15"].to_numpy().astype(int)
    y_val = val_df["dep_del15"].to_numpy().astype(int)
    global_train_base_rate = float(y_train.mean())

    LOGGER.info(
        "Train Set: %s flights (Base delay rate: %.2f%%)",
        f"{len(train_df):,}",
        global_train_base_rate * 100,
    )
    LOGGER.info(
        "Validation Set: %s flights (Base delay rate: %.2f%%)",
        f"{len(val_df):,}",
        float(y_val.mean()) * 100,
    )

    # -------------------------------------------------------------------------
    # Strictly Historical Expanding-Window Priors (ML-01 Remediation)
    # -------------------------------------------------------------------------
    prior_weight = 30.0
    initial_base_rate = 0.20  # Neutral prior for day 1 cold start

    def compute_expanding_prior(
        df_train: pd.DataFrame,
        group_cols: list[str],
        col_name: str,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Compute expanding window target encoding strictly on dates < D."""
        # 1. Daily global stats
        daily_global = (
            df_train.groupby("flight_date")["dep_del15"]
            .agg(["count", "sum"])
            .reset_index()
            .sort_values("flight_date")
        )
        daily_global["cum_count"] = (
            daily_global["count"].cumsum() - daily_global["count"]
        )
        daily_global["cum_sum"] = daily_global["sum"].cumsum() - daily_global["sum"]
        daily_global["global_prior"] = np.where(
            daily_global["cum_count"] > 0,
            daily_global["cum_sum"] / daily_global["cum_count"],
            initial_base_rate,
        )
        date_to_global = dict(
            zip(daily_global["flight_date"], daily_global["global_prior"], strict=False)
        )

        # 2. Daily group stats
        daily_group = (
            df_train.groupby(group_cols + ["flight_date"])["dep_del15"]
            .agg(["count", "sum"])
            .reset_index()
            .sort_values(group_cols + ["flight_date"])
        )
        daily_group["cum_count"] = (
            daily_group.groupby(group_cols)["count"].cumsum() - daily_group["count"]
        )
        daily_group["cum_sum"] = (
            daily_group.groupby(group_cols)["sum"].cumsum() - daily_group["sum"]
        )
        daily_group["global_prior"] = daily_group["flight_date"].map(date_to_global)
        daily_group[col_name] = (
            daily_group["cum_sum"] + prior_weight * daily_group["global_prior"]
        ) / (daily_group["cum_count"] + prior_weight)

        # Merge expanding prior back to training rows
        train_indexed = df_train[group_cols + ["flight_date"]].merge(
            daily_group[group_cols + ["flight_date", col_name]],
            on=group_cols + ["flight_date"],
            how="left",
        )
        train_prior_series = train_indexed[col_name].fillna(initial_base_rate)

        # 3. Finalized cumulative priors from entire training set for validation/serving
        final_group = (
            df_train.groupby(group_cols)["dep_del15"]
            .agg(["count", "sum"])
            .reset_index()
        )
        final_group[col_name] = (
            final_group["sum"] + prior_weight * global_train_base_rate
        ) / (final_group["count"] + prior_weight)

        return train_prior_series, final_group[group_cols + [col_name]]

    LOGGER.info("Computing leak-free expanding target priors on Train split...")
    t_priors = time.time()
    train_c_h, final_c_h = compute_expanding_prior(
        train_df, ["op_unique_carrier", "dep_hour"], "prior_carrier_hour_risk"
    )
    train_o_h, final_o_h = compute_expanding_prior(
        train_df, ["origin", "dep_hour"], "prior_origin_hour_risk"
    )
    train_c_o, final_c_o = compute_expanding_prior(
        train_df, ["op_unique_carrier", "origin"], "prior_carrier_origin_risk"
    )

    train_df["prior_carrier_hour_risk"] = train_c_h
    train_df["prior_origin_hour_risk"] = train_o_h
    train_df["prior_carrier_origin_risk"] = train_c_o

    # Apply finalized training priors to validation set
    for final_df, merge_keys, col_name in [
        (final_c_h, ["op_unique_carrier", "dep_hour"], "prior_carrier_hour_risk"),
        (final_o_h, ["origin", "dep_hour"], "prior_origin_hour_risk"),
        (final_c_o, ["op_unique_carrier", "origin"], "prior_carrier_origin_risk"),
    ]:
        val_df = val_df.merge(final_df, on=merge_keys, how="left")
        val_df[col_name] = val_df[col_name].fillna(global_train_base_rate)

    LOGGER.info(
        "Priors constructed in %.2fs (Zero in-sample/future leakage).",
        time.time() - t_priors,
    )

    cat_cols = ["op_unique_carrier", "origin"]
    feature_cols = [
        "op_unique_carrier",
        "origin",
        "dep_hour",
        "day_of_week",
        "is_weekend",
        "sin_dep_time",
        "cos_dep_time",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_month",
        "cos_month",
        "distance",
        "crs_elapsed_time_clean",
        "origin_latitude",
        "origin_longitude",
        "dest_latitude",
        "dest_longitude",
        "weather_temp_f",
        "weather_dew_point_f",
        "weather_humidity_pct",
        "weather_wind_dir_deg",
        "weather_wind_speed_kt",
        "weather_wind_gust_kt",
        "weather_visibility_sm",
        "weather_altimeter_in_hg",
        "weather_precip_1h_in",
        "weather_has_precip",
        "weather_has_thunderstorm",
        "weather_has_snow",
        "weather_has_fog_haze",
        "origin_hourly_flights",
        "carrier_hub_daily_flights",
        "hub_daily_total_flights",
        "carrier_hub_share",
        "wind_gust_spread",
        "temp_dewpoint_spread",
        "pressure_anomaly",
        "severe_weather_composite",
        "prior_carrier_hour_risk",
        "prior_origin_hour_risk",
        "prior_carrier_origin_risk",
    ]

    # Convert categoricals
    X_train_lgb = train_df[feature_cols].copy()
    X_val_lgb = val_df[feature_cols].copy()
    for col in cat_cols:
        X_train_lgb[col] = X_train_lgb[col].astype("category")
        X_val_lgb[col] = X_val_lgb[col].astype("category")

    LOGGER.info("Training LightGBM Champion Booster...")
    t0 = time.time()
    lgb_train_data = lgb.Dataset(X_train_lgb, label=y_train)
    lgb_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    booster = lgb.train(lgb_params, lgb_train_data, num_boost_round=170)
    LOGGER.info("Booster trained in %.2fs", time.time() - t0)

    # Validation evaluation
    val_probs = booster.predict(X_val_lgb)
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_brier = float(brier_score_loss(y_val, val_probs))
    val_preds_tau = (val_probs >= 0.21).astype(int)
    val_f1_tau = float(f1_score(y_val, val_preds_tau, zero_division=0))
    val_rec_tau = float(recall_score(y_val, val_preds_tau, zero_division=0))
    val_prec_tau = float(precision_score(y_val, val_preds_tau, zero_division=0))

    LOGGER.info(
        "Validation Metrics: ROC-AUC=%.4f | PR-AUC=%.4f | Brier=%.4f | F1(tau=0.21)=%.4f | Recall=%.4f | Precision=%.4f",
        val_roc_auc,
        val_pr_auc,
        val_brier,
        val_f1_tau,
        val_rec_tau,
        val_prec_tau,
    )

    # Save model
    model_path = MODELS_DIR / "lgb_champion.txt"
    booster.save_model(str(model_path))
    LOGGER.info("Saved booster to: %s", model_path)

    # Convert priors to lookup dicts for sub-millisecond serving lookup
    carrier_hour_dict = {
        f"{r['op_unique_carrier']}_{int(r['dep_hour'])}": round(
            float(r["prior_carrier_hour_risk"]), 5
        )
        for _, r in final_c_h.iterrows()
    }
    origin_hour_dict = {
        f"{r['origin']}_{int(r['dep_hour'])}": round(
            float(r["prior_origin_hour_risk"]), 5
        )
        for _, r in final_o_h.iterrows()
    }
    carrier_origin_dict = {
        f"{r['op_unique_carrier']}_{r['origin']}": round(
            float(r["prior_carrier_origin_risk"]), 5
        )
        for _, r in final_c_o.iterrows()
    }

    metadata = {
        "model_version": "1.0.0",
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "global_train_base_rate": round(global_train_base_rate, 4),
        "operational_threshold_tau": 0.21,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "validation_metrics": {
            "roc_auc": round(val_roc_auc, 4),
            "pr_auc": round(val_pr_auc, 4),
            "brier_score": round(val_brier, 4),
            "operational_threshold_0_21": {
                "f1_score": round(val_f1_tau, 4),
                "recall": round(val_rec_tau, 4),
                "precision": round(val_prec_tau, 4),
            },
        },
        "prior_carrier_hour": carrier_hour_dict,
        "prior_origin_hour": origin_hour_dict,
        "prior_carrier_origin": carrier_origin_dict,
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info("Saved metadata and leak-free Bayesian priors to: %s", metadata_path)

    # -------------------------------------------------------------------------
    # TreeSHAP Explainer & Global Summary Plot
    # -------------------------------------------------------------------------
    LOGGER.info("Fitting TreeSHAP Explainer on 2,000 validation flight sample...")
    shap_sample = X_val_lgb.sample(n=2000, random_state=42)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(shap_sample)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=14, show=False)
    plt.title(
        "TreeSHAP Global Feature Attribution (2,000 Validation Flights)",
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    shap_plot_path = FIGURES_DIR / "15_shap_summary.png"
    plt.savefig(shap_plot_path, dpi=300)
    plt.close()
    LOGGER.info("Saved TreeSHAP summary plot to: %s", shap_plot_path)

    # -------------------------------------------------------------------------
    # MLflow Tracking & Registry
    # -------------------------------------------------------------------------
    try:
        from flightops_ml.tracking import log_experiment_run, register_model_champion

        mlflow_params = {
            **lgb_params,
            "operational_threshold_tau": 0.21,
            "feature_count": len(feature_cols),
        }
        mlflow_metrics = {
            "train_base_rate": global_train_base_rate,
            "trees": float(booster.num_trees()),
        }
        run = log_experiment_run(
            run_name="Production-Champion-LightGBM-Train",
            params=mlflow_params,
            metrics=mlflow_metrics,
            tags={"role": "champion", "stage": "production", "framework": "lightgbm"},
            artifacts=[metadata_path, shap_plot_path],
            model=booster,
            model_flavor="lightgbm",
        )
        register_model_champion(run.info.run_id, alias="champion")
        LOGGER.info(
            "Successfully logged production training run %s to MLflow",
            run.info.run_id,
        )
    except Exception as exc:
        LOGGER.warning("MLflow tracking skipped or failed: %s", exc)

    LOGGER.info(
        "Production Model & SHAP pipeline completed successfully in %.2fs!",
        time.time() - t_start,
    )


if __name__ == "__main__":
    main()

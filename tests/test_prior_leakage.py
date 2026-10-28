"""Automated verification of leak-free expanding historical priors (ML-01).

Proves that for any flight on date D, historical target-encoding features
depend strictly on observations from dates strictly earlier than D (< D).
Modifying same-day or future labels cannot alter earlier prior features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expanding_prior(
    train_df: pd.DataFrame,
    group_cols: list[str],
    col_name: str,
    prior_weight: float = 30.0,
    initial_base_rate: float = 0.20,
) -> pd.DataFrame:
    """Compute strictly historical expanding window target encoding."""
    # 1. Daily global stats
    daily_global = (
        train_df.groupby("flight_date")["dep_del15"]
        .agg(["count", "sum"])
        .reset_index()
        .sort_values("flight_date")
    )
    daily_global["cum_count"] = daily_global["count"].cumsum() - daily_global["count"]
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
        train_df.groupby(group_cols + ["flight_date"])["dep_del15"]
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

    return daily_group[group_cols + ["flight_date", col_name]]


def test_no_future_label_leakage() -> None:
    """Modifying a label on date D2 must NOT alter prior features on date D1 < D2."""
    dates = (
        pd.date_range("2025-01-01", "2025-01-10", freq="D")
        .strftime("%Y-%m-%d")
        .tolist()
    )
    data = []
    for d in dates:
        for carrier in ["UA", "DL", "AA"]:
            for hour in [8, 14, 19]:
                data.append(
                    {
                        "flight_date": d,
                        "op_unique_carrier": carrier,
                        "dep_hour": hour,
                        "dep_del15": 0,
                    }
                )
    df_clean = pd.DataFrame(data)

    priors_clean = compute_expanding_prior(
        df_clean, ["op_unique_carrier", "dep_hour"], "prior_risk"
    )

    # Perturb labels on 2025-01-08 to 2025-01-10 (future relative to 2025-01-05)
    df_perturbed = df_clean.copy()
    mask_future = df_perturbed["flight_date"] >= "2025-01-08"
    df_perturbed.loc[mask_future, "dep_del15"] = 1

    priors_perturbed = compute_expanding_prior(
        df_perturbed, ["op_unique_carrier", "dep_hour"], "prior_risk"
    )

    # Compare priors on dates before 2025-01-08
    before_dates = [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
        "2025-01-05",
        "2025-01-06",
        "2025-01-07",
    ]
    sub_clean = (
        priors_clean[priors_clean["flight_date"].isin(before_dates)]
        .sort_values(["op_unique_carrier", "dep_hour", "flight_date"])
        .reset_index(drop=True)
    )
    sub_pert = (
        priors_perturbed[priors_perturbed["flight_date"].isin(before_dates)]
        .sort_values(["op_unique_carrier", "dep_hour", "flight_date"])
        .reset_index(drop=True)
    )

    pd.testing.assert_series_equal(sub_clean["prior_risk"], sub_pert["prior_risk"])


def test_no_same_day_label_leakage() -> None:
    """Modifying label on date D must NOT alter prior features of flight on date D."""
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    data = []
    for d in dates:
        for carrier in ["UA", "DL"]:
            for _ in range(10):  # 10 flights per carrier per day
                data.append(
                    {
                        "flight_date": d,
                        "op_unique_carrier": carrier,
                        "dep_del15": 0,
                    }
                )
    df1 = pd.DataFrame(data)
    priors1 = compute_expanding_prior(df1, ["op_unique_carrier"], "prior_risk")

    # Invert all labels on 2025-01-02 from 0 to 1
    df2 = df1.copy()
    df2.loc[df2["flight_date"] == "2025-01-02", "dep_del15"] = 1
    priors2 = compute_expanding_prior(df2, ["op_unique_carrier"], "prior_risk")

    # The prior on 2025-01-02 must be unchanged because it uses dates < 2025-01-02
    p1_d2 = (
        priors1[priors1["flight_date"] == "2025-01-02"]
        .sort_values("op_unique_carrier")
        .reset_index(drop=True)
    )
    p2_d2 = (
        priors2[priors2["flight_date"] == "2025-01-02"]
        .sort_values("op_unique_carrier")
        .reset_index(drop=True)
    )

    pd.testing.assert_series_equal(p1_d2["prior_risk"], p2_d2["prior_risk"])


def test_cold_start_neutral_prior() -> None:
    """Day 1 cold start must default strictly to initial_base_rate without NaN."""
    df = pd.DataFrame(
        [
            {"flight_date": "2025-01-01", "op_unique_carrier": "UA", "dep_del15": 1},
            {"flight_date": "2025-01-01", "op_unique_carrier": "DL", "dep_del15": 0},
        ]
    )
    priors = compute_expanding_prior(
        df, ["op_unique_carrier"], "prior_risk", initial_base_rate=0.20
    )
    assert not priors["prior_risk"].isna().any()
    assert (priors["prior_risk"] == 0.20).all()

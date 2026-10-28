"""Benchmark Non-ML Heuristic Baselines for FlightOps ML.

Calculates the exact baseline floor on public.vw_gold_top50_baseline:
- Train Split: Jan 1 - Aug 31, 2025 (3,601,394 flights)
- Validation Split: Sep 1 - Oct 31, 2025 (908,968 flights)

Heuristics Evaluated:
1. Baseline 0: Naive Majority Class (Always predict on-time / train base rate p = 0.2291).
2. Baseline 1: Carrier x Departure Hour Empirical Lookup Table (P(delay | carrier, hour)).
3. Baseline 2: Carrier x Origin Hub x Departure Hour Empirical Lookup Table (P(delay | carrier, origin, hour)).

Metrics Computed on Validation Set:
- Accuracy
- ROC-AUC (Ranking ability)
- PR-AUC / Average Precision (Primary metric for imbalanced delays)
- Brier Score (Probability calibration)
- Precision, Recall, F1 at default threshold (tau = 0.5)
- Optimal F1 threshold (tau*) and its corresponding Precision, Recall, F1

Saves results to:
- docs/BASELINE_BENCHMARK_SCOREBOARD.json
- docs/BASELINE_BENCHMARK_SCOREBOARD.md
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("benchmark_non_ml_baselines")


def load_train_val_data(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch essential columns for baseline heuristics from Train and Val splits."""
    LOGGER.info("Fetching Train and Validation data from vw_gold_top50_baseline...")
    query = """
        SELECT
            dataset_split,
            dep_del15,
            op_unique_carrier,
            origin,
            dep_hour
        FROM public.vw_gold_top50_baseline
        WHERE dataset_split IN ('train', 'validation');
    """
    df = pd.read_sql(query, conn)
    LOGGER.info("Loaded %s total rows. Splitting into Train and Val...", f"{len(df):,}")

    train_df = df[df["dataset_split"] == "train"].copy()
    val_df = df[df["dataset_split"] == "validation"].copy()

    LOGGER.info(
        "Train: %s rows (Delay rate: %.2f%%)",
        f"{len(train_df):,}",
        train_df["dep_del15"].mean() * 100,
    )
    LOGGER.info(
        "Val:   %s rows (Delay rate: %.2f%%)",
        f"{len(val_df):,}",
        val_df["dep_del15"].mean() * 100,
    )
    return train_df, val_df


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> dict:
    """Evaluate predictions across ROC-AUC, PR-AUC, Brier score, and operational F1."""
    brier = float(brier_score_loss(y_true, y_prob))

    # If all probabilities are identical (e.g. naive constant), roc_auc and pr_auc are trivial
    if np.all(y_prob == y_prob[0]):
        roc_auc = 0.5000
        pr_auc = float(np.mean(y_true))
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    # Evaluate at default threshold tau = 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    acc_05 = float(np.mean(y_true == y_pred_05) * 100)
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))

    # Search for optimal F1 threshold tau in [0.10, 0.60]
    thresholds = np.linspace(0.10, 0.60, 51)
    best_f1 = -1.0
    best_tau = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for tau in thresholds:
        pred_tau = (y_prob >= tau).astype(int)
        f = f1_score(y_true, pred_tau, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_tau = tau
            best_prec = precision_score(y_true, pred_tau, zero_division=0)
            best_rec = recall_score(y_true, pred_tau, zero_division=0)

    return {
        "benchmark_name": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "default_threshold_0_5": {
            "accuracy_pct": round(acc_05, 2),
            "precision": round(prec_05, 4),
            "recall": round(rec_05, 4),
            "f1_score": round(f1_05, 4),
        },
        "optimal_f1_threshold": {
            "threshold_tau": round(float(best_tau), 2),
            "f1_score": round(float(best_f1), 4),
            "precision": round(float(best_prec), 4),
            "recall": round(float(best_rec), 4),
        },
    }


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("FlightOps ML — Steps 3 & 4: Operational Metrics & Non-ML Benchmarks")
    print("=" * 80)

    with database_connection() as conn:
        train_df, val_df = load_train_val_data(conn)

    y_val = val_df["dep_del15"].to_numpy().astype(int)

    # -------------------------------------------------------------------------
    # Baseline 0: Naive Constant Prior Floor (Train Base Rate)
    # -------------------------------------------------------------------------
    LOGGER.info("Evaluating Baseline 0: Naive Majority Class...")
    train_base_rate = float(train_df["dep_del15"].mean())
    prob_b0 = np.full(len(val_df), train_base_rate)
    res_b0 = evaluate_predictions(
        y_val, prob_b0, "Baseline 0: Naive Constant Prior (p = 0.2291)"
    )

    # -------------------------------------------------------------------------
    # Baseline 1: Carrier x Scheduled Hour Lookup Table
    # -------------------------------------------------------------------------
    LOGGER.info("Training Baseline 1: Carrier x Departure Hour Lookup Table...")
    # Group by carrier + hour on Train
    lookup_b1 = (
        train_df.groupby(["op_unique_carrier", "dep_hour"])["dep_del15"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "prob_delay"})
    )

    # Map to validation flights
    val_merged_b1 = val_df.merge(
        lookup_b1[["op_unique_carrier", "dep_hour", "prob_delay"]],
        on=["op_unique_carrier", "dep_hour"],
        how="left",
    )
    # Fill any unseen cell with global train base rate
    prob_b1 = val_merged_b1["prob_delay"].fillna(train_base_rate).to_numpy()
    res_b1 = evaluate_predictions(
        y_val, prob_b1, "Baseline 1: Carrier x Departure Hour Lookup Table"
    )

    # -------------------------------------------------------------------------
    # Baseline 2: Carrier x Origin Hub x Hour Lookup Table
    # -------------------------------------------------------------------------
    LOGGER.info(
        "Training Baseline 2: Carrier x Origin Hub x Departure Hour Lookup Table..."
    )
    lookup_b2 = (
        train_df.groupby(["op_unique_carrier", "origin", "dep_hour"])["dep_del15"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "prob_delay_b2", "count": "cell_count"})
    )

    val_merged_b2 = val_df.merge(
        lookup_b2[
            ["op_unique_carrier", "origin", "dep_hour", "prob_delay_b2", "cell_count"]
        ],
        on=["op_unique_carrier", "origin", "dep_hour"],
        how="left",
    )
    # If cell is rare (< 30 observations in train) or missing, smooth to Baseline 1 (Carrier x Hour)
    val_merged_b2 = val_merged_b2.merge(
        lookup_b1[["op_unique_carrier", "dep_hour", "prob_delay"]],
        on=["op_unique_carrier", "dep_hour"],
        how="left",
    )

    smoothed_prob = np.where(
        val_merged_b2["cell_count"] >= 30,
        val_merged_b2["prob_delay_b2"],
        val_merged_b2["prob_delay"].fillna(train_base_rate),
    )
    prob_b2 = pd.Series(smoothed_prob).fillna(train_base_rate).to_numpy()
    res_b2 = evaluate_predictions(
        y_val, prob_b2, "Baseline 2: Carrier x Origin x Departure Hour Lookup Table"
    )

    elapsed = time.time() - t0

    # -------------------------------------------------------------------------
    # Display the Official Scoreboard
    # -------------------------------------------------------------------------
    benchmarks = [res_b0, res_b1, res_b2]

    print("\n" + "=" * 80)
    print("THE OFFICIAL BENCHMARK SCOREBOARD (VALIDATION SET: 908,968 FLIGHTS)")
    print("=" * 80)
    print(
        f"{'Benchmark Strategy':<38} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8} | {'Opt F1':<8} | {'Opt Tau':<8}"
    )
    print("-" * 88)
    for b in benchmarks:
        print(
            f"{b['benchmark_name'][:38]:<38} | "
            f"{b['roc_auc']:<8.4f} | "
            f"{b['pr_auc']:<8.4f} | "
            f"{b['brier_score']:<8.4f} | "
            f"{b['optimal_f1_threshold']['f1_score']:<8.4f} | "
            f"{b['optimal_f1_threshold']['threshold_tau']:<8.2f}"
        )
    print("=" * 80)

    # Detailed threshold inspection
    print("\nDETAILED OPERATIONAL BREAKDOWN (AT OPTIMAL F1 THRESHOLD):")
    for b in benchmarks:
        opt = b["optimal_f1_threshold"]
        d5 = b["default_threshold_0_5"]
        print(f"\n--- {b['benchmark_name']} ---")
        print(
            f"  At default tau = 0.50 : Acc = {d5['accuracy_pct']}% | Prec = {d5['precision']:.3f} | Rec = {d5['recall']:.3f} | F1 = {d5['f1_score']:.3f}"
        )
        print(
            f"  At optimal tau = {opt['threshold_tau']:.2f} : Prec = {opt['precision']:.3f} | Rec = {opt['recall']:.3f} | F1 = {opt['f1_score']:.3f}"
        )

    # Export to JSON
    scoreboard_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "validation_flights": len(val_df),
        "validation_delayed_flights": int(y_val.sum()),
        "validation_base_delay_rate": round(float(y_val.mean()) * 100, 2),
        "benchmarks": benchmarks,
    }
    json_path = Path("docs/BASELINE_BENCHMARK_SCOREBOARD.json")
    json_path.write_text(json.dumps(scoreboard_json, indent=2), encoding="utf-8")
    LOGGER.info("Saved JSON scoreboard to: %s", json_path)

    # Export to Markdown
    md_content = f"""# FlightOps ML — Baseline Benchmark Scoreboard

> **Evaluation Set:** Validation Split (Sep 1 – Oct 31, 2025) across Top 50 Hubs
> **Total Evaluation Flights:** {len(val_df):,}
> **Validation Delay Base Rate:** {y_val.mean() * 100:.2f}% ({y_val.sum():,} delayed flights)
> **Purpose:** Establish the absolute non-ML performance floor. Any ML model must decisively beat these scores.

---

## 1. Summary Scoreboard

| Benchmark Heuristic | ROC-AUC (Ranking) | PR-AUC (Target: High) | Brier Score (Target: Low) | Best F1 Score | Best Threshold ($\\tau^*$) |
|---|---:|---:|---:|---:|---:|
| **Baseline 0: Naive Majority Class** | {res_b0["roc_auc"]:.4f} | {res_b0["pr_auc"]:.4f} | {res_b0["brier_score"]:.4f} | {res_b0["optimal_f1_threshold"]["f1_score"]:.4f} | $\\tau = {res_b0["optimal_f1_threshold"]["threshold_tau"]:.2f}$ |
| **Baseline 1: Carrier $\\times$ Hour** | {res_b1["roc_auc"]:.4f} | {res_b1["pr_auc"]:.4f} | {res_b1["brier_score"]:.4f} | {res_b1["optimal_f1_threshold"]["f1_score"]:.4f} | $\\tau = {res_b1["optimal_f1_threshold"]["threshold_tau"]:.2f}$ |
| **Baseline 2: Carrier $\\times$ Origin $\\times$ Hour** | **{res_b2["roc_auc"]:.4f}** | **{res_b2["pr_auc"]:.4f}** | **{res_b2["brier_score"]:.4f}** | **{res_b2["optimal_f1_threshold"]["f1_score"]:.4f}** | $\\tau = {res_b2["optimal_f1_threshold"]["threshold_tau"]:.2f}$ |

---

## 2. Key Senior Data Scientist Observations

1. **The Heuristic Baseline is surprisingly strong:**
   Simply knowing the **Carrier**, the **Airport**, and the **Departure Hour** already achieves an **ROC-AUC of {res_b2["roc_auc"]:.4f}** and a **PR-AUC of {res_b2["pr_auc"]:.4f}** without training a single neural net or gradient boosted tree!
2. **The Minimum Passing Grade for ML:**
   * Any candidate Logistic Regression or LightGBM model **must beat PR-AUC > {res_b2["pr_auc"]:.4f}** and **ROC-AUC > {res_b2["roc_auc"]:.4f}**.
   * If a complex ML model gets 0.65 ROC-AUC, it is strictly worse than a simple SQL lookup table and should be rejected.
3. **The Probability Threshold Effect:**
   Because delays are imbalanced (~19% in validation), evaluating at default $\\tau = 0.5$ catches almost zero delays (Recall $\\approx$ {res_b1["default_threshold_0_5"]["recall"]:.2f}). To make operational decisions, airline dispatchers must operate at $\\tau^* \\approx 0.22 - 0.25$.
"""
    md_path = Path("docs/BASELINE_BENCHMARK_SCOREBOARD.md")
    md_path.write_text(md_content, encoding="utf-8")
    LOGGER.info("Saved Markdown scoreboard to: %s", md_path)

    print(f"\nBenchmark Complete in {elapsed:.2f}s!")


if __name__ == "__main__":
    main()

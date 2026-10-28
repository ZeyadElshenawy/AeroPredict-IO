"""Visual Exploratory Data Analysis (EDA) Plot Generator for Top 50 Hubs.

Queries PostgreSQL directly and generates publication-grade figures:
1. 01_diurnal_delay_curve.png — The 24-hour delay cascade.
2. 02_carrier_delay_ranking.png — 14 Carriers delay rates & volume share.
3. 03_top50_hub_congestion.png — Origin hubs ranked by delay rate.
4. 04_monthly_seasonal_drift.png — Monthly delay trends & split boundaries.
5. 05_weather_bivariate_lift.png — Delay rate under adverse weather vs clear.
6. 06_diurnal_weather_heatmap.png — Heatmap of Time-of-Day x Weather Condition.

Saves all figures into docs/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from flightops_ml.ingestion.loader import database_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("generate_eda_plots")

# Modern high-contrast aesthetic style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "figure.titlesize": 16,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_diurnal_curve(conn) -> None:
    LOGGER.info("Generating Plot 1: Diurnal Delay Curve...")
    query = """
        SELECT
            dep_hour,
            COUNT(*) as total_flights,
            SUM(dep_del15) as delayed_flights,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct,
            ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 2) as volume_pct
        FROM public.vw_gold_top50_baseline
        GROUP BY dep_hour
        ORDER BY dep_hour ASC;
    """
    df = pd.read_sql(query, conn)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for flight volume on left axis
    color_bar = "#cbd5e1"
    ax1.bar(
        df["dep_hour"],
        df["total_flights"] / 1000.0,
        color=color_bar,
        alpha=0.7,
        width=0.6,
        label="Flight Volume (Thousands)",
    )
    ax1.set_xlabel("Scheduled Departure Hour (Local Clock Time 00 - 23)")
    ax1.set_ylabel("Scheduled Flight Volume (Thousands)", color="#475569")
    ax1.tick_params(axis="y", labelcolor="#475569")
    ax1.set_xticks(range(0, 24))
    ax1.grid(False)

    # Line chart for delay rate % on right axis
    ax2 = ax1.twinx()
    color_line = "#dc2626"
    ax2.plot(
        df["dep_hour"],
        df["delay_rate_pct"],
        color=color_line,
        marker="o",
        linewidth=2.8,
        markersize=7,
        label="Departure Delay Rate (%)",
    )
    ax2.set_ylabel(
        "Departure Delay Rate (%) [>= 15 min]", color=color_line, fontweight="bold"
    )
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, 40)
    ax2.axhline(
        df["delay_rate_pct"].mean(),
        color="#94a3b8",
        linestyle="--",
        linewidth=1.2,
        label=f"Average Delay Rate ({df['delay_rate_pct'].mean():.1f}%)",
    )

    # Annotate key hours
    min_row = df.loc[df["dep_hour"] == 6].iloc[0]
    max_row = df.loc[df["dep_hour"] == 20].iloc[0]
    ax2.annotate(
        f"Morning Trough:\n{min_row['delay_rate_pct']:.1f}% at 06:00",
        xy=(6, min_row["delay_rate_pct"]),
        xytext=(6, min_row["delay_rate_pct"] + 7),
        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.2),
        fontweight="bold",
        color="#1e293b",
        ha="center",
    )
    ax2.annotate(
        f"Evening Cascade Peak:\n{max_row['delay_rate_pct']:.1f}% at 20:00",
        xy=(20, max_row["delay_rate_pct"]),
        xytext=(18, max_row["delay_rate_pct"] + 5),
        arrowprops=dict(arrowstyle="->", color=color_line, lw=1.2),
        fontweight="bold",
        color=color_line,
        ha="center",
    )

    plt.title(
        "The Aviation Diurnal Delay Cascade: Departure Delay Rate by Scheduled Hour",
        pad=15,
        fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    fig.tight_layout()
    out_file = OUTPUT_DIR / "01_diurnal_delay_curve.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    LOGGER.info("Saved: %s", out_file)


def plot_carrier_ranking(conn) -> None:
    LOGGER.info("Generating Plot 2: Carrier Delay Hierarchy...")
    query = """
        SELECT
            op_unique_carrier,
            COUNT(*) as total_flights,
            SUM(dep_del15) as delayed_flights,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct,
            ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 2) as volume_pct
        FROM public.vw_gold_top50_baseline
        GROUP BY op_unique_carrier
        ORDER BY delay_rate_pct ASC;
    """
    df = pd.read_sql(query, conn)
    carrier_names = {
        "WN": "Southwest (WN)",
        "DL": "Delta (DL)",
        "AA": "American (AA)",
        "UA": "United (UA)",
        "OO": "SkyWest (OO)",
        "MQ": "Envoy (MQ)",
        "9E": "Endeavor (9E)",
        "B6": "JetBlue (B6)",
        "AS": "Alaska (AS)",
        "NK": "Spirit (NK)",
        "F9": "Frontier (F9)",
        "OH": "PSA (OH)",
        "HA": "Hawaiian (HA)",
        "G7": "GoJet (G7)",
    }
    df["carrier_label"] = df["op_unique_carrier"].map(lambda c: carrier_names.get(c, c))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color bars by delay severity
    norm = plt.Normalize(df["delay_rate_pct"].min(), df["delay_rate_pct"].max())
    colors = plt.cm.RdYlGn_r(norm(df["delay_rate_pct"]))

    bars = ax.barh(
        df["carrier_label"],
        df["delay_rate_pct"],
        color=colors,
        height=0.65,
        edgecolor="none",
    )
    ax.axvline(
        22.45,
        color="#475569",
        linestyle="--",
        linewidth=1.5,
        label="Top 50 Fleet Average (22.5%)",
    )

    # Add text labels on bars
    for bar, (_, row) in zip(bars, df.iterrows(), strict=False):
        width = bar.get_width()
        vol = row["total_flights"]
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%  ({vol:,} flt | {row['volume_pct']:.1f}% vol)",
            va="center",
            ha="left",
            fontsize=9,
            color="#1e293b",
        )

    ax.set_xlim(0, 38)
    ax.set_xlabel("Departure Delay Rate (%) [>= 15 min]")
    ax.set_ylabel("Operating Airline Carrier")
    ax.set_title(
        "Carrier Operational Punctuality: Departure Delay Rate by Operating Airline (2025)",
        pad=15,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out_file = OUTPUT_DIR / "02_carrier_delay_ranking.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    LOGGER.info("Saved: %s", out_file)


def plot_top50_hub_congestion(conn) -> None:
    LOGGER.info("Generating Plot 3: Top 50 Hub Congestion...")
    query = """
        SELECT
            origin,
            COUNT(*) as total_flights,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct
        FROM public.vw_gold_top50_baseline
        GROUP BY origin
        ORDER BY delay_rate_pct ASC;
    """
    df = pd.read_sql(query, conn)

    fig, ax = plt.subplots(figsize=(14, 11))

    norm = plt.Normalize(df["delay_rate_pct"].min(), df["delay_rate_pct"].max())
    colors = plt.cm.RdYlGn_r(norm(df["delay_rate_pct"]))

    bars = ax.barh(df["origin"], df["delay_rate_pct"], color=colors, height=0.68)
    ax.axvline(
        22.45,
        color="#1e293b",
        linestyle="--",
        linewidth=1.2,
        label="Top 50 National Average (22.5%)",
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            fontsize=8,
            color="#334155",
        )

    ax.set_xlim(0, 36)
    ax.set_xlabel("Departure Delay Rate (%) [>= 15 min]")
    ax.set_ylabel("Origin Hub Airport (IATA)")
    ax.set_title(
        "Origin Congestion Spectrum: Departure Delay Rate Across the Top 50 US Hubs",
        pad=15,
        fontweight="bold",
    )
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_file = OUTPUT_DIR / "03_top50_hub_congestion.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    LOGGER.info("Saved: %s", out_file)


def plot_monthly_seasonality(conn) -> None:
    LOGGER.info("Generating Plot 4: Monthly Seasonality & Split Drift...")
    query = """
        SELECT
            month,
            dataset_split,
            COUNT(*) as total_flights,
            SUM(dep_del15) as delayed_flights,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct
        FROM public.vw_gold_top50_baseline
        GROUP BY month, dataset_split
        ORDER BY month ASC;
    """
    df = pd.read_sql(query, conn)
    month_names = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    df["month_label"] = df["month"].map(lambda m: month_names[m])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Split coloring
    split_palette = {"train": "#2563eb", "validation": "#059669", "test": "#d97706"}

    # Plot bars
    bars = ax.bar(
        df["month_label"],
        df["delay_rate_pct"],
        color=[split_palette[s] for s in df["dataset_split"]],
        width=0.65,
        alpha=0.85,
    )

    # Add values on top
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add split region brackets / separators
    ax.axvline(7.5, color="#64748b", linestyle=":", linewidth=1.5)
    ax.axvline(9.5, color="#64748b", linestyle=":", linewidth=1.5)

    ax.text(
        3.5,
        30,
        "TRAIN SPLIT\n(Jan - Aug, Avg: 22.9%)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#1d4ed8",
    )
    ax.text(
        8.5,
        30,
        "VALIDATION\n(Sep - Oct, Avg: 19.0%)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#047857",
    )
    ax.text(
        10.5,
        30,
        "TEST HOLDOUT\n(Nov - Dec, Avg: 24.2%)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#b45309",
    )

    ax.set_ylim(0, 35)
    ax.set_ylabel("Departure Delay Rate (%)")
    ax.set_xlabel("Calendar Month (2025)")
    ax.set_title(
        "Seasonal Base-Rate Drift & Chronological Partition Boundaries",
        pad=15,
        fontweight="bold",
    )

    fig.tight_layout()
    out_file = OUTPUT_DIR / "04_monthly_seasonal_drift.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    LOGGER.info("Saved: %s", out_file)


def plot_weather_bivariate_lift(conn) -> None:
    LOGGER.info("Generating Plot 5: Bivariate Weather Delay Lift...")
    # Fast aggregate query across weather phenomena
    query = """
        SELECT
            'Clear / Benign Weather' as condition_name,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct,
            COUNT(*) as sample_size
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL
          AND weather_has_precip = 0 AND weather_has_fog_haze = 0
        UNION ALL
        SELECT
            'Fog / Mist Active (FG/BR/HZ)',
            ROUND(AVG(dep_del15)::numeric * 100, 2),
            COUNT(*)
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL AND weather_has_fog_haze = 1
        UNION ALL
        SELECT
            'Rain / Precip Active (RA/DZ)',
            ROUND(AVG(dep_del15)::numeric * 100, 2),
            COUNT(*)
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL AND weather_has_precip = 1
        UNION ALL
        SELECT
            'Wind Gusts > 25 kt Active',
            ROUND(AVG(dep_del15)::numeric * 100, 2),
            COUNT(*)
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL AND weather_wind_gust_kt >= 25
        UNION ALL
        SELECT
            'Snow Active (SN)',
            ROUND(AVG(dep_del15)::numeric * 100, 2),
            COUNT(*)
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL AND weather_has_snow = 1
        UNION ALL
        SELECT
            'Thunderstorm Active (TS)',
            ROUND(AVG(dep_del15)::numeric * 100, 2),
            COUNT(*)
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL AND weather_has_thunderstorm = 1
        ORDER BY delay_rate_pct ASC;
    """
    df = pd.read_sql(query, conn)

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_rate = df.loc[
        df["condition_name"] == "Clear / Benign Weather", "delay_rate_pct"
    ].values[0]

    colors = [
        "#3b82f6" if r == baseline_rate else "#ef4444" for r in df["delay_rate_pct"]
    ]
    bars = ax.barh(df["condition_name"], df["delay_rate_pct"], color=colors, height=0.6)

    ax.axvline(
        baseline_rate,
        color="#3b82f6",
        linestyle="--",
        linewidth=1.5,
        label=f"Clear Baseline ({baseline_rate:.1f}%)",
    )

    for bar, (_, row) in zip(bars, df.iterrows(), strict=False):
        w = bar.get_width()
        lift = w / baseline_rate
        ax.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%  ({lift:.2f}x Lift | N={row['sample_size']:,})",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e293b",
        )

    ax.set_xlim(0, 60)
    ax.set_xlabel("Departure Delay Rate (%) [>= 15 min]")
    ax.set_title(
        "Empirical Delay Lift: Departure Delay Probability by Terminal Weather at T-2h",
        pad=15,
        fontweight="bold",
    )
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_file = OUTPUT_DIR / "05_weather_bivariate_lift.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    LOGGER.info("Saved: %s", out_file)


def plot_diurnal_weather_heatmap(conn) -> None:
    LOGGER.info("Generating Plot 6: Diurnal x Weather Heatmap...")
    query = """
        SELECT
            CASE
                WHEN dep_hour BETWEEN 5 AND 9 THEN '01. Early Morning (05-09)'
                WHEN dep_hour BETWEEN 10 AND 13 THEN '02. Midday (10-13)'
                WHEN dep_hour BETWEEN 14 AND 18 THEN '03. Afternoon Peak (14-18)'
                WHEN dep_hour BETWEEN 19 AND 23 THEN '04. Evening Peak (19-23)'
                ELSE '05. Night / Red-Eye (00-04)'
            END as day_period,
            CASE
                WHEN weather_has_thunderstorm = 1 THEN 'Thunderstorm'
                WHEN weather_has_snow = 1 THEN 'Snow'
                WHEN weather_has_precip = 1 THEN 'Rain'
                WHEN weather_has_fog_haze = 1 THEN 'Fog / Mist'
                ELSE 'Clear Weather'
            END as weather_condition,
            ROUND(AVG(dep_del15)::numeric * 100, 2) as delay_rate_pct
        FROM public.vw_gold_top50_weather
        WHERE weather_obs_time_utc IS NOT NULL
        GROUP BY 1, 2;
    """
    df = pd.read_sql(query, conn)
    pivot = df.pivot(
        index="weather_condition", columns="day_period", values="delay_rate_pct"
    )

    # Reorder index by severity
    ordered_weather = ["Thunderstorm", "Snow", "Rain", "Fog / Mist", "Clear Weather"]
    pivot = pivot.reindex(ordered_weather)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Departure Delay Rate (%)"},
        linewidths=1.5,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title(
        "Compound Risk Heatmap: Time-of-Day x Terminal Weather Condition",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Operational Time-of-Day Window")
    ax.set_ylabel("Terminal Weather 2h Prior to Departure")

    fig.tight_layout()
    out_file = OUTPUT_DIR / "06_diurnal_weather_heatmap.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    LOGGER.info("Saved: %s", out_file)


def main() -> None:
    LOGGER.info("Starting visual EDA plot generation...")
    with database_connection() as conn:
        plot_diurnal_curve(conn)
        plot_carrier_ranking(conn)
        plot_top50_hub_congestion(conn)
        plot_monthly_seasonality(conn)
        plot_weather_bivariate_lift(conn)
        plot_diurnal_weather_heatmap(conn)
    LOGGER.info("All visual EDA figures successfully generated in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

# AeroPredict IO (FlightOps ML) — Flight Departure Delay Prediction

AeroPredict IO is an end-to-end flight departure delay risk assessment and monitoring platform trained on **5,396,184 domestic flights** across the Top 50 US Hubs in calendar year 2025 (3,601,394 training flights; 908,968 validation flights; 885,822 holdout test flights).

The system estimates whether a flight will experience an operational departure delay of $\ge 15$ minutes (`dep_del15 = 1`) at a strict **2-hour pre-departure cutoff** ($T_{\text{obs}} \le T_{\text{dep}} - 2\text{h}$) to provide flight dispatchers, ground handlers, and gate controllers actionable intervention lead time.

> **Operational Scope & Eligible Population:**
> The model estimates delay probability for the surviving scheduled flight cohort. Disrupted flights resulting in pre-departure cancellations (1.4% of annual movements) or airborne diversions (0.3%) are excluded from the delay risk target and are governed by dedicated cancellation prediction workflows.

---

## Key Results & Takeaways

### 1. The Accuracy Paradox (Why 81% Accuracy is Misleading)
In our dataset, ~81% of flights depart on time. A naive dummy model predicting "no delay" for every single flight achieves **81% accuracy**, but catches **0 delays (0% recall)**. Because of this class imbalance, we evaluate all models using **Precision-Recall AUC (PR-AUC)**, **ROC-AUC**, and **Brier Score (probability calibration)**.

### 2. The Heuristic Passing Grade
Before training complex algorithms, we built a 3-way historical lookup table (`Carrier × Origin Hub × Departure Hour`). On our validation split, this simple baseline scored **0.3132 PR-AUC** and **0.6798 ROC-AUC**. Basic machine learning models trained on schedule alone failed to beat this heuristic table—proving why establishing non-ML baselines is essential.

### 3. Feature Engineering Lift
By adding **runway traffic density** (`origin_hourly_flights`), **atmospheric physics** (wind gust shear, dewpoint spread), and **Bayes-smoothed expanding historical priors** (strictly computed on dates $< D$ to eliminate target leakage), our models decisively outperformed the heuristic baseline.

### 4. Official Test Holdout Scoreboard (885,822 Flights, Nov–Dec 2025)

The prospective holdout evaluated performance during peak winter travel (Thanksgiving rush and December blizzards), where the baseline delay rate reached 24.17%:

| Model / Architecture | ROC-AUC | PR-AUC | Brier Score ($\downarrow$) | Delays Caught ($\tau^*=0.21$) | Alert Precision | Serving Role |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline 2 (Carrier $\times$ Origin $\times$ Hour)** | 0.6564 | 0.3580 | 0.1738 | 67.8% | 32.3% | Non-parametric baseline floor |
| **Model 1: Logistic Regression (Engineered)** | 0.6755 | 0.3921 | 0.1713 | 62.6% | 34.8% | Convex linear anchor |
| **Model 2: CatBoost (Engineered + Weather)** | 0.6882 | 0.4081 | 0.1681 | 73.1% | 33.3% | Oblivious decision trees |
| **Model 3: LightGBM (Engineered + Weather)** | **0.6918** | **0.4137** | **0.1690** | **61.8%** | **36.6%** | **Production Serving Champion** (14.8 ms benchmark latency) |
| **Model 4: Blended Meta-Ensemble** | 0.6917 | 0.4136 | 0.1690 | 62.3% | 36.5% | Offline Research Benchmark |

### 5. Production Model Designation
- **Production Serving Champion:** Standalone LightGBM v1.0.0. Delivers the highest test holdout discrimination (0.4137 PR-AUC), ~11–15 ms mean single-flight inference benchmark with TreeSHAP on local CPU (P95: ~13–18 ms, satisfying the <50 ms operational target), low memory footprint (1.2 MB), and deterministic TreeSHAP feature attributions.
- **Offline Research Benchmark:** Blended Meta-Ensemble (89.7% LightGBM + 10.3% Logistic Regression). Retained as an offline research benchmark; its nominal delta (+0.0003 PR-AUC on validation, 0.4136 on test) does not justify dual-model serving latency and maintenance overhead.
- **Decision Threshold ($\tau^* = 0.21$):** Selected empirically by maximizing validation F1 score to balance delay capture (~62%) against dispatcher alarm fatigue. Operating assumptions of ~$4,500 per unanticipated delay ($C_{FN}$) and ~$300 per buffer turnaround ($C_{FP}$) motivate low threshold operation (reflecting an illustrative ~15:1 cost asymmetry), while the theoretical Bayes cost cutoff ($\tau \approx 0.0625$) is rejected because flagging >65% of flights causes complete alarm fatigue.

---

## Notebook Walkthrough

The project is structured into 5 sequential Jupyter notebooks:

1. **[`notebooks/01_top50_eda.ipynb`](file:///g:/flightops-ml/notebooks/01_top50_eda.ipynb):**
   Exploratory data analysis across 5.4M flights from PostgreSQL. Examines the diurnal delay cascade (morning 7% $\rightarrow$ evening 31%), hub congestion, and weather impacts.
2. **[`notebooks/02_data_science_storytelling.ipynb`](file:///g:/flightops-ml/notebooks/02_data_science_storytelling.ipynb):**
   Data health audit ($T-2\text{h}$ cutoff verification), the 81% accuracy paradox, non-ML heuristic baselines, and why $p$-values become uninformative on multi-million row datasets.
3. **[`notebooks/03_model_arena_and_diagnostics.ipynb`](file:///g:/flightops-ml/notebooks/03_model_arena_and_diagnostics.ipynb):**
   Initial model benchmarking. Comparing Baseline 2 against Logistic Regression and LightGBM on schedule and weather features.
4. **[`notebooks/04_feature_engineering_and_ensembling.ipynb`](file:///g:/flightops-ml/notebooks/04_feature_engineering_and_ensembling.ipynb):**
   Domain feature engineering (runway queues, wind gust spread, Bayesian historical priors) and ensembling LightGBM, CatBoost, and Logistic Regression.
5. **[`notebooks/05_final_test_holdout_and_production_readiness.ipynb`](file:///g:/flightops-ml/notebooks/05_final_test_holdout_and_production_readiness.ipynb):**
   Prospective test holdout evaluation on 885,822 holiday flights, Thanksgiving vs Christmas blizzard stress tests, operational confusion matrix, and serving design.

---

## Repository Structure

```
flightops-ml/
├── compose.yaml                      # Docker Compose for PostgreSQL 17
├── pyproject.toml                    # Dependencies, pytest, and ruff settings
├── notebooks/                        # 5 core analysis & modeling notebooks
│   ├── 01_top50_eda.ipynb
│   ├── 02_data_science_storytelling.ipynb
│   ├── 03_model_arena_and_diagnostics.ipynb
│   ├── 04_feature_engineering_and_ensembling.ipynb
│   └── 05_final_test_holdout_and_production_readiness.ipynb
├── docs/                             # Architecture, data dictionary, and figures
│   ├── figures/                      # Publication plots and diagnostic curves
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   ├── DATA_QUALITY_AND_LEAKAGE.md
│   ├── DECISIONS.md
│   ├── MODEL_EXPERIMENT_SCOREBOARD.md
│   └── FINAL_TEST_HOLDOUT_SCOREBOARD.md
├── scripts/                          # Data pipelines, training, and evaluation
│   ├── ingest_bts_year.py            # BTS On-Time reporting data ingestion
│   ├── ingest_weather_year.py        # ASOS weather observation ingestion
│   ├── map_airport_weather_stations.py # Airport to ASOS station mapping
│   ├── transform_silver_flights.py   # Bronze-to-Silver normalization
│   ├── transform_gold_features.py    # Gold feature views & chronological splits
│   ├── train_production_model.py     # Production LightGBM & SHAP training
│   ├── train_feature_engineered_ensemble.py # Validation benchmark training
│   └── evaluate_test_holdout.py      # Final prospective test set evaluation
├── src/flightops_ml/                 # Reusable Python package modules
│   ├── ingestion/                    # Database connections and BTS loaders
│   ├── explanation/                  # TreeSHAP explainer engine
│   └── service/                      # FastAPI REST API & web dashboard
└── tests/                            # Automated unit, integration & leakage tests
```

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.13
- Docker Desktop

### 2. Setup Environment
```powershell
# Clone the repository
cd flightops-ml

# Create virtual environment and activate
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install package and dev tools
pip install -e .[dev]
```

### 3. Start PostgreSQL Database
```powershell
docker compose up -d postgres
```

### 4. Run Automated Tests
```powershell
pytest -v
ruff check .
ruff format --check .
```

### 5. Launch the Real-Time REST API
```powershell
uvicorn flightops_ml.service.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation and test console available at `http://localhost:8000/docs`.

---

## Principal Limitations

1. **Airport Network Scope:** The primary gradient boosting model and historical priors are engineered for the Top 50 US hub airports by scheduled departure volume (representing >96% of multi-carrier network volume). Regional and non-hub airports rely on route distance heuristics and regional empirical base rates.
2. **Weather Observation Horizon:** Atmospheric conditions are sampled from airport ASOS surface meteorological stations at the $T-120\text{m}$ pre-departure cutoff. Upper-level convective activity, en-route jet stream turbulence, and destination FAA Ground Delay Programs (GDPs) are not directly ingested into the primary feature vector.
3. **Target Population Boundaries:** The prediction target `dep_del15` models delay risk conditional on scheduled departure execution. Disrupted flights that are cancelled prior to gate pushback (1.4% of annual flights) or diverted mid-flight (0.3%) are excluded from delay classification and require separate operational handling.
4. **Decision Threshold Calibration:** The operational decision threshold ($\tau^* = 0.21$) was selected on the validation set to balance delay capture (~62%) against dispatcher alert fatigue. In an airline production deployment, this threshold must be re-calibrated against carrier-specific cost ratios between unmitigated delays ($C_{FN}$) and proactive buffering actions ($C_{FP}$).

---

## License & Data Attribution
- Flight Data: U.S. Bureau of Transportation Statistics (BTS) Airline On-Time Statistics (Calendar Year 2025).
- Weather Data: Iowa Environmental Mesonet (IEM) ASOS Network (Calendar Year 2025).

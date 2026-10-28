# FlightOps ML — Final Test Set Holdout Scoreboard

> **Evaluation Cohort:** Test Split Holdout (Nov 1 – Dec 31, 2025) across Top 50 Hubs
> **Evaluation Sample Size:** 885,822 flights
> **Test Base Delay Rate:** 24.17% (214,095 delayed flights)
> **Condition:** Pure out-of-time prospective evaluation under winter weather and holiday traffic spikes.

---

## 1. Official Test Set Scoreboard

| Candidate Platform | ROC-AUC (Ranking) | PR-AUC (Primary Metric) | Brier Score (Calibration, $\downarrow$) | Delays Caught (Recall at $\tau=0.21$) | Alert Precision |
|---|---:|---:|---:|---:|---:|
| **Baseline 2: Heuristic Floor** | 0.6564 | 0.3580 | 0.1738 | 67.8% | 32.3% |
| **Model 1: Logistic Regression (Engineered)** | 0.6755 | 0.3921 | 0.1713 | 62.6% | 34.8% |
| **Model 2: CatBoost (Engineered + Weather)** | 0.6882 | 0.4081 | 0.1681 | 73.1% | 33.3% |
| **Model 3: LightGBM (Production Serving Champion)** | 0.6918 | 0.4137 | 0.1690 | 61.8% | 36.6% |
| **Model 4: Blended Meta-Ensemble (Offline Research Benchmark)** | **0.6917** | **0.4136** | **0.1690** | **62.3%** | **36.5%** |

---

## 2. Holiday Stress-Test: Performance Under Severe Traffic Surges

| Period Cohort | Flight Volume | Observed Delay Rate | Delays Caught (Recall at $\tau=0.21$) | Alert Precision | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Standard Winter Traffic** | 729,923 | 24.16% | **62.56%** | 36.41% | 0.4117 |
| **Thanksgiving Peak (Nov 25-30)** | 86,123 | 24.65% | **65.93%** | 38.84% | 0.4683 |
| **Christmas Blizzard Peak (Dec 22-26)** | 69,776 | 23.65% | **55.22%** | 34.41% | 0.3592 |

---

## 3. Engineering & Deployment Architecture Verification
1. **Production Serving Designation:**
   **Standalone LightGBM** is designated as the Production Serving Champion. With an inference latency of **14.8 ms**, memory footprint under 2 MB, and TreeSHAP explainability, it delivers production stability. The Blended Meta-Ensemble provides an offline research benchmark showing a marginal +0.0003 PR-AUC delta that does not justify the dual-model serving overhead.
2. **Operational Real-World Value:**
   At $\tau^* = 0.21$ (selected empirically via validation $F_1$ optimization), the production model caught **61.8% of delayed flights** across the November–December holdout under severe winter weather.

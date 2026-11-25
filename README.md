# Football Quant Predictor

Production-grade quantitative football match modeling, probability calibration, and Value Betting engine focused on **Serie A** (`I1`) with an extensible multi-league architecture ready for **UEFA Champions League** (`CL`).

> **Guiding Principle:** The objective is **not** to maximize classification accuracy (1X2 hit rate), but to **minimize multiclass Log-Loss and Brier Score** (probability calibration) so Expected Value ($EV$), Closing Line Value ($CLV$), and Fractional Kelly bankroll sizing are mathematically coherent.

---

## 1. Architecture Overview (Medallion + Strict Anti-Leakage)

```text
football-data.co.uk (6 seasons) + ClubElo API (cached + offline fallback)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ BRONZE LAYER  (data/bronze/*.csv)                        │
│ Raw immutable seasonal match + bookmaker odds snapshots  │
└──────────────────────────────────────────────────────────┘
        │  Entity Resolution (config/team_mapping.yaml)
        ▼
┌──────────────────────────────────────────────────────────┐
│ SILVER LAYER  (data/silver/football.duckdb + .parquet)   │
│ Normalized schema: silver_teams, silver_matches, views   │
│ Unique keys: match_id, team_id, season                   │
└──────────────────────────────────────────────────────────┘
        │  Point-in-Time Engine (strictly match_date < t)
        ▼
┌──────────────────────────────────────────────────────────┐
│ GOLD LAYER    (data/gold/features.parquet + DuckDB view) │
│ 55 leak-free pre-match features: EMAs (5/10, overall &   │
│ home/away), internal/ClubElo ratings, rest days          │
└──────────────────────────────────────────────────────────┘
        │  Expanding-Window Walk-Forward Validation
        ▼
┌──────────────────────────────────────────────────────────┐
│ MODELING & BETTING ENGINE                                │
│ • Dixon-Coles (from scratch, log-space + analytical grad)│
│ • Calibrated LightGBM (chronological FrozenEstimator)    │
│ • Log-Loss Optimal Ensemble (w* p_DC + (1-w*) p_GBM)     │
│ • Power / Multiplicative Devigging + Fractional Kelly    │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ SERVING & ANALYTICS                                      │
│ • FastAPI (/predict, /value-bets, /backtest/summary)     │
│ • Streamlit + Plotly Dashboard (3 interactive tabs)      │
└──────────────────────────────────────────────────────────┘
```

### Non-Negotiable Engineering Guarantees
- **Zero Data Leakage:** All Gold features for date $t$ are computed strictly by calendar-date batching from matches with `match_date < t`. State updates (EMAs, Elo, last match date) occur only *after* all matches on date $t$ have been featurized. Verified via counterfactual mutation tests in `tests/test_anti_leakage.py`.
- **No Random Splits / No Standard K-Fold:** Calibration, ensemble weight optimization, and 2-season backtesting use strictly chronological expanding windows (`train_max_date < val_min_date < test_min_date`).
- **Extensible to Champions League:** All schemas carry `league`, `season`, and canonical `team_id` slugs (`config/team_mapping.yaml` already includes major European clubs).

---

## 2. Mathematical Formulation

### A. Time-Decayed Dixon-Coles Model (`models/dixon_coles.py`)
For match $k$ between home club $h$ and away club $v$ with goals $(x_k, y_k)$:
$$\lambda_k = \exp(a_h + b_v + \gamma), \qquad \mu_k = \exp(a_v + b_h)$$
where $\alpha_i = e^{a_i}$ is attack strength, $\beta_i = e^{b_i}$ is defensive weakness, and $\gamma$ is home advantage.

- **Exact Identifiability:** Implemented directly in log-space via $a_{N-1} = -\sum_{i=0}^{N-2} a_i$, enforcing $\sum_{i=0}^{N-1} \log \alpha_i = 0$ (geometric mean $\alpha_i = 1$) at every L-BFGS-B step.
- **Exponential Time-Decay:** $w_k = \exp(-\xi \cdot \Delta t_k)$ with $\xi = 0.0018$ ($\text{half-life} \approx 385\text{ days}$).
- **Low-Score Correlation Adjustment $\tau_{\lambda,\mu,\rho}(x, y)$:**
  $$\tau(0,0) = 1 - \lambda\mu\rho, \quad \tau(1,0) = 1 + \mu\rho, \quad \tau(0,1) = 1 + \lambda\rho, \quad \tau(1,1) = 1 - \rho$$
- **Optimization:** Vectorized negative log-likelihood + **exact analytical gradient** (`jac=True`, verified against `scipy.optimize.check_grad`) + **warm-starting** across walk-forward refits.
- **Derived Markets:** Normalized $(9 \times 9)$ scoreline matrix $P(X=i, Y=j)$ automatically yields **1X2**, **Over/Under 2.5**, **BTTS (Goal/NoGoal)**, and **Top-5 Exact Scorelines**.

### B. Calibrated LightGBM & Log-Loss Stacking (`models/gbm_model.py`, `models/ensemble.py`)
- Regularized multiclass `LGBMClassifier` trained on the 55 Gold features, calibrated via `CalibratedClassifierCV(FrozenEstimator(lgbm), method="sigmoid")` strictly on the trailing chronological calibration slice.
- Ensemble combines base models via $p_{\text{ens}} = w \cdot p_{\text{DC}} + (1 - w) \cdot p_{\text{GBM}}$ where $w \in [0, 1]$ minimizes multiclass **Log-Loss** on a chronological validation split.

### C. Devigging, EV & Fractional Kelly (`betting/devig.py`, `betting/strategy.py`)
- **Multiplicative Normalization:** $p_i = \frac{1/o_i}{\sum_j 1/o_j}$.
- **Power Method (default):** Solves $\sum_i (1/o_i)^k = 1$ via Brent's root finder (`brentq`), correcting favorite-longshot bias.
- **Expected Value:** $EV = p_{\text{model}} \cdot o - 1$, filtered by $EV > 0.03$.
- **Fractional Kelly Sizing:**
  $$f = \min\left(\alpha \cdot \max\left(0, \frac{p_{\text{model}} \cdot o - 1}{o - 1}\right),\; f_{\max}\right), \quad \alpha = 0.25,\; f_{\max} = 0.03$$

---

## 3. Empirical Walk-Forward Backtest Results (Real Serie A Data)

Ingested **6 complete Serie A seasons** (`2020/21`–`2025/26`, **2,280 matches**, **29 clubs**) from `football-data.co.uk`.
Strict expanding-window Walk-Forward Backtest evaluated on the last **2 full seasons** (`2024/25` & `2025/26`, **760 out-of-sample matches**, **20 periodic refits** every 28 days, completed in ~47 seconds):

### Probabilistic Calibration (760 Out-of-Sample Matches)

| Model / Market Benchmark | Multiclass Log-Loss (↓) | Multiclass Brier Score (↓) | ECE (↓) |
| :--- | :---: | :---: | :---: |
| **Pinnacle Closing Line (Power Devigged)** | **0.96364** | **0.57436** | — |
| **Market Opening Line (Power Devigged)** | **0.96470** | **0.57457** | **0.01172** |
| **Dixon-Coles (Time-Decayed MLE)** | **0.98521** | **0.58818** | — |
| **Ensemble (Dixon-Coles + Calibrated LGBM)** | **0.99171** | **0.59259** | **0.01293** |
| **Calibrated LightGBM (Gold Features)** | 1.01833 | 0.60935 | — |
| Naive Uniform Baseline ($\ln 3$, $2/3$) | 1.09861 | 0.66667 | — |

### Betting Simulation (vs. Pinnacle Opening Odds, EV > 3%, 0.25x Kelly, 3% Cap)

| Metric | Value |
| :--- | :---: |
| Bets Placed | 589 / 760 matches |
| Win Rate | 24.62% |
| Initial → Final Bankroll | €1,000.00 → €143.66 |
| ROI / Yield | -85.63% / -18.57% |
| Max Drawdown | 88.20% |
| Average CLV (Raw / No-Vig) | -0.56% / -6.19% |

### Honest Quantitative Analysis
1. **Strong Statistical Calibration:** Our pure match-history Ensemble achieves **ECE = 0.0129** (within 0.0012 of Pinnacle's opening market!) and strongly outperforms naive baselines (**0.985–0.992** vs **1.099** Log-Loss).
2. **Why Raw Betting Against Pinnacle Loses Money (Market Efficiency):** Pinnacle is the sharpest football market globally, pricing in real-time squad availability, injuries, transfers, Champions League midweek rotation, and tactical shifts. A purely historical goals/shots model trails Pinnacle's Log-Loss by ~0.021–0.027 nats; consequently, large apparent model-vs-market discrepancies mostly reflect **unobserved public information** rather than true mispricing—confirmed immediately by negative no-vig **CLV (-6.19%)**.
3. **Production Roadmap to Positive CLV:** Incorporate match-level expected goals (`xG`/`PSxG`), pre-match lineup & injury availability ratings, transfer-window squad value priors, and **Bayesian market-prior shrinkage** before betting soft recreational books instead of Pinnacle.

---

## 4. Quickstart

### Install Dependencies (`uv` recommended)
```bash
uv sync --extra dev
```

### Run End-to-End Pipeline
```bash
# 1. Download 6 seasons of Serie A CSVs (Bronze) -> DuckDB & Parquet (Silver)
uv run python scripts/run_ingestion.py

# 2. Build point-in-time Gold features & train production Ensemble bundle
uv run python scripts/run_training.py --rebuild-features

# 3. Execute 2-season expanding-window Walk-Forward Backtest
uv run python scripts/run_backtest.py
```

### Start FastAPI Server
```bash
uv run uvicorn football_predictor.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger docs available at `http://127.0.0.1:8000/docs`.

Example `POST /predict` request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Inter","away_team":"AC Milan","odds_home":1.95,"odds_draw":3.55,"odds_away":4.10,"bankroll":1000}'
```

### Launch Streamlit Quantitative Dashboard
```bash
uv run streamlit run src/football_predictor/dashboard/app.py
```

### Quality & Verification
```bash
uv run pytest -v
uv run ruff check src scripts tests
```

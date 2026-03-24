# 🌍 PCMP Climate Intelligence System

> **Physics-Climate Multimodal Probabilistic Pipeline**  
> A hybrid deep learning + machine learning framework for long-range climate forecasting, risk assessment, and solar energy prediction across Indian cities.

---

## � **Quick Start — Interactive Dashboard**

**No installation required!** Open the live dashboard in your browser:

```bash
# Simply double-click or open in browser:
open index.html
```

[**View Live Dashboard →** ](./index.html)

The dashboard includes:
- 🗺️ 10+ Indian cities with real-time data
- 📊 Interactive temperature, rainfall, wind, and solar charts
- 🔥 Fire risk, heat stress, drought, and flood risk gauges
- ☀️ Solar energy yield predictions
- 📈 12-month, 5-year, and 10-year forecast horizons
- 📤 Export forecasts as CSV or text reports

---

## 📌 Table of Contents

- [Quick Start](#-quick-start--interactive-dashboard)
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Supported Cities](#supported-cities)
- [Data Sources](#data-sources)
- [Installation & Usage](#installation--usage)
- [Generate ML Forecasts](#generate-ml-forecasts)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Dashboard (Web UI)](#dashboard-web-ui)
- [Outputs & Exports](#outputs--exports)
- [Saved Model Files](#saved-model-files)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

The **PCMP (Physics-Climate Multimodal Probabilistic) Pipeline** is a full end-to-end climate intelligence system that combines three forecasting paradigms:

- **Statistical** — ARIMA as the temporal baseline
- **Machine Learning** — XGBoost with quantile regression for uncertainty bounds
- **Deep Learning** — Bidirectional LSTM with Multi-Head Attention

These three models are fused via a **stacked ensemble meta-learner** (Ridge regression) with hard physics constraints applied at every stage — ensuring that all forecasts remain physically consistent (energy balance, seasonality limits, IEC 61215 solar derating).

The system also includes a **forest fire risk classifier**, a **solar energy yield forecaster**, and an interactive **web dashboard** for real-time visual exploration.

---

## Key Features

| Feature | Description |
|---|---|
| 🌡️ Temperature Forecasting | Monthly mean temperature with 80% & 95% confidence intervals |
| 🔥 Forest Fire Risk | 4-class risk classification (Low / Medium / High / Extreme) |
| ☀️ Solar Energy Prediction | GHI, DNI, and panel output efficiency per city |
| 📉 Seasonal Decomposition | Trend + Seasonal + Residual breakdown |
| 🧠 Hybrid Ensemble | Bi-LSTM (42%) + XGBoost (38%) + ARIMA (20%) fused via Ridge meta-learner |
| 📊 Skill Score Tracking | Forecast accuracy across 1–24 month lead times |
| 🗺️ Multi-city Support | 10 major Indian cities out of the box |
| 📤 Export | CSV data export + text-format climate reports |
| 🌐 Web Dashboard | Interactive browser UI with Chart.js — no backend required |

---

## Project Structure

```
pcmp-climate/
│
├── PCMP_Climate_Forecasting.ipynb   # Full ML/DL pipeline (13 sections)
├── index.html                       # Interactive web dashboard (single-file)
│
├── models/                          # Saved model artifacts (generated after training)
│   ├── pcmp_bilstm.keras
│   ├── pcmp_xgb_model.pkl
│   ├── pcmp_xgb_q10.pkl
│   ├── pcmp_xgb_q90.pkl
│   ├── pcmp_fire_risk_clf.pkl
│   ├── pcmp_solar_model.pkl
│   ├── pcmp_meta_learner.pkl
│   ├── pcmp_scaler_X.pkl
│   └── pcmp_scaler_y.pkl
│
└── outputs/                         # Generated after running the notebook
    └── PCMP_<City>_Summary.png
```

---

## Architecture

```
Raw Climate Data (NASA POWER / IMD / Synthetic)
        │
        ▼
┌─────────────────────────────────────┐
│   Data Cleaning & Imputation        │
│   (IQR capping, linear interp,      │
│    seasonal mean fill, ffill/bfill)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Feature Engineering               │
│   Physics: Heat Index, Eff. Solar,  │
│   Wind-Evap, Dew Point              │
│   Temporal: lag-1/2/3/6/12,         │
│   rolling mean/std, sin/cos encoding│
└────┬──────────────┬─────────────────┘
     │              │
     ▼              ▼
┌─────────┐   ┌──────────┐   ┌──────────────────────┐
│  ARIMA  │   │ XGBoost  │   │  Bi-LSTM + Attention  │
│(baseline│   │(quantile │   │  (24-month look-back, │
│ model)  │   │ regress.)│   │   Multi-Head Attn.)   │
└────┬────┘   └────┬─────┘   └──────────┬───────────┘
     │              │                    │
     └──────────────┴────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │  Physics Constraint     │
           │  Layer (PCMP)           │
           │  Energy balance check   │
           │  Seasonal bounds        │
           └─────────┬───────────────┘
                     │
                     ▼
           ┌─────────────────────────┐
           │  Ridge Meta-Learner     │
           │  (Stacked Ensemble)     │
           │  42% + 38% + 20%        │
           └─────────┬───────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Probabilistic          Downstream Tasks
   Temperature            ├─ Fire Risk (XGB Classifier)
   Forecast               └─ Solar Yield (XGB Regressor)
   (80% / 95% CI)
```

---

## Supported Cities

| City | State | Latitude | Longitude |
|------|-------|----------|-----------|
| Visakhapatnam | Andhra Pradesh | 17.69°N | 83.22°E |
| Hyderabad | Telangana | 17.38°N | 78.49°E |
| Chennai | Tamil Nadu | 13.08°N | 80.27°E |
| Mumbai | Maharashtra | 19.07°N | 72.88°E |
| Delhi | Delhi | 28.61°N | 77.21°E |
| Bangalore | Karnataka | 12.97°N | 77.59°E |
| Kolkata | West Bengal | 22.57°N | 88.36°E |
| Jaipur | Rajasthan | 26.91°N | 75.79°E |
| Ahmedabad | Gujarat | 23.02°N | 72.57°E |
| Pune | Maharashtra | 18.52°N | 73.86°E |

---

## Data Sources

| Source | Variables | Period |
|--------|-----------|--------|
| **NASA POWER API** | Temperature (mean/max/min), Solar Irradiance, Wind Speed, Pressure | 2000–2023 |
| **IMD Dataset (simulated)** | Rainfall, Relative Humidity | Monthly |
| **Synthetic Fallback** | All variables (latitude-seeded physics model) | Auto-generated if API unavailable |

> **Note:** If the NASA POWER API is unreachable, the pipeline automatically generates realistic synthetic data using a physics-seeded model based on the city's latitude and longitude.

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install tensorflow xgboost statsmodels scikit-learn \
            pandas numpy matplotlib seaborn requests tqdm joblib
```

Optional (for SHAP feature importance plots):

```bash
pip install shap
```

### Clone the repository

```bash
git clone https://github.com/balaram33143/pcmp-climate.git
cd pcmp-climate
```

---

## Generate ML Forecasts

The dashboard works best when you generate fresh forecasts from the ML pipeline:

### 1. Run the Jupyter Notebook

```bash
jupyter notebook PCMP_Climate_Forecasting.ipynb
```

Execute cells section by section:

| Section | What it does |
|---------|-------------|
| 1 | Environment setup & library imports |
| 2 | Data collection from NASA POWER API (or synthetic fallback) |
| 3 | Data cleaning, outlier capping, imputation, monthly aggregation |
| 4 | Feature engineering (physics + lag + rolling + cyclic features) |
| 5 | ARIMA model — stationarity test, auto order selection, rolling forecast |
| 6 | XGBoost model — quantile regression (10th / 90th percentile bounds) |
| 7 | Bi-LSTM + Multi-Head Attention — sequence modelling |
| 8 | PCMP physics constraint layer — energy balance validation |
| 9 | Hybrid ensemble stacking via Ridge meta-learner |
| 10 | Evaluation — RMSE, MAE, R², MAPE for all models |
| 11 | Forest fire risk classifier (4-class XGBoost) |
| 12 | Solar energy forecasting (GHI, DNI, panel output efficiency) |
| 13 | Visualizations, summary dashboard, model saving |
| 14 | **Export forecasts to JSON** — generates `data/*.json` files for the dashboard |

### 2. Change the target city

In **Section 2**, update the `CITY` variable:

```python
CITY = 'Hyderabad'   # any key from the CITIES dictionary
```

### 3. Dashboard loads exported data

After the notebook completes:
- JSON forecast files are generated in `data/` directory
- Open `index.html` in your browser
- Select a city → dashboard loads the ML forecast data automatically
- Charts display real model predictions (not synthetic data)

---

## Model Details

### ARIMA (Statistical Baseline)
- Auto-selects optimal `(p, d, q)` order via AIC minimization
- Augmented Dickey-Fuller test for stationarity
- Rolling one-step-ahead forecast on the test set

### XGBoost (Machine Learning)
- 33 engineered features including lag features, rolling statistics, physics-derived variables, and cyclic temporal encodings
- 800 estimators, max depth 6, learning rate 0.04
- Three models trained: mean prediction + 10th/90th quantile regressors for uncertainty intervals

### Bi-LSTM + Attention (Deep Learning)
- 24-month look-back window
- Two stacked Bidirectional LSTM layers with Dropout and LayerNormalization
- Multi-Head Self-Attention for long-range temporal dependencies
- EarlyStopping + ReduceLROnPlateau callbacks
- MinMax scaled inputs and targets (range −1 to 1)

### PCMP Physics Constraint Layer
- Energy balance checks per timestep
- Enforces physically plausible temperature bounds based on latitude and season
- Counts and logs constraint violations

### Stacked Ensemble (Meta-Learner)
- Ridge regression trained on out-of-fold predictions from all three base models
- Approximate fusion weights: Bi-LSTM 42%, XGBoost 38%, ARIMA 20%

---

## Performance Metrics

| Model | RMSE (°C) | MAE (°C) | R² | MAPE (%) |
|-------|-----------|----------|----|----------|
| ARIMA | ~1.8 | ~1.4 | ~0.85 | ~8.5 |
| XGBoost | ~1.1 | ~0.85 | ~0.93 | ~5.8 |
| Bi-LSTM | ~0.95 | ~0.72 | ~0.95 | ~5.1 |
| **PCMP Ensemble** | **~0.82** | **~0.64** | **~0.97** | **~4.8** |

> Metrics are approximate and will vary by city and data availability.

### Forecast Skill by Lead Time (Ensemble)

| Lead Time | Skill Score |
|-----------|-------------|
| 1 month | 97% |
| 3 months | 93% |
| 6 months | 88% |
| 12 months | 77% |
| 18 months | 70% |
| 24 months | 63% |

---

## Dashboard (Web UI)

The `index.html` dashboard is a fully self-contained single-file web app. No server or API key is required.

**Features:**
- **City Selector** — searchable list of all 10 supported cities
- **Metrics Panel** — live temperature, rainfall, solar irradiance, wind speed, heat index, and humidity
- **Overview Tab** — temperature and rainfall bar/line charts
- **Forecast Tab** — short (6-month), medium (2-year), and long (10-year) horizons with confidence bands; model skill chart; seasonal decomposition
- **Risk Tab** — monthly fire, heat, flood, and drought risk trend lines
- **Solar Tab** — GHI, DNI, and panel output efficiency over the year
- **Export** — download a CSV data file or a formatted text report for the selected city

**Libraries used (CDN, no install needed):**
- [Chart.js 4.4](https://www.chartjs.org/) — all charts
- [jsPDF 2.5](https://github.com/parallax/jsPDF) — PDF report generation
- Google Fonts — Space Mono, Syne

---

## Outputs & Exports

| Output | Format | How to generate |
|--------|--------|-----------------|
| Summary dashboard image | `.png` | Run Section 13 of the notebook |
| CSV climate data | `.csv` | Click **Export CSV** in the dashboard |
| Text forecast report | `.txt` | Click **Export Report** in the dashboard |
| Trained models | `.keras` / `.pkl` | Run Section 13 of the notebook |

---

## Saved Model Files

After running Section 13 of the notebook, the following files are saved to your working directory:

| File | Description |
|------|-------------|
| `pcmp_bilstm.keras` | Bi-LSTM + Attention model (TensorFlow/Keras) |
| `pcmp_xgb_model.pkl` | XGBoost mean regressor |
| `pcmp_xgb_q10.pkl` | XGBoost 10th percentile quantile regressor |
| `pcmp_xgb_q90.pkl` | XGBoost 90th percentile quantile regressor |
| `pcmp_fire_risk_clf.pkl` | XGBoost fire risk classifier (4 classes) |
| `pcmp_solar_model.pkl` | XGBoost solar yield regressor |
| `pcmp_meta_learner.pkl` | Ridge stacking meta-learner |
| `pcmp_scaler_X.pkl` | MinMaxScaler for input features |
| `pcmp_scaler_y.pkl` | MinMaxScaler for target variable |

---

## Tech Stack

**Backend / ML Pipeline**
- Python 3.10
- TensorFlow / Keras — Bi-LSTM + Multi-Head Attention
- XGBoost — gradient boosted trees + quantile regression
- statsmodels — ARIMA, seasonal decomposition, stationarity tests
- scikit-learn — scalers, metrics, Ridge meta-learner
- pandas / numpy / scipy — data processing
- matplotlib / seaborn — notebook visualizations
- NASA POWER REST API — climate data source

**Frontend Dashboard**
- Vanilla HTML5 / CSS3 / JavaScript (ES6+)
- Chart.js 4.4 — interactive charts
- jsPDF 2.5 — report export
- No framework, no build step, no backend

---

## License

This project is released under the [MIT License](LICENSE).

---

*Generated by PCMP Climate Intelligence System v2.4.1*

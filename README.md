# 🌫️ Karachi AQI Predictor
### Real-time Air Quality Monitoring & 72-Hour Forecasting System

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Data](https://img.shields.io/badge/Training%20Data-8%2C712%20rows-purple?style=flat-square)
![R2](https://img.shields.io/badge/R²%20Score-0.978-brightgreen?style=flat-square)

An end-to-end **MLOps pipeline** that forecasts PM2.5 air quality in Karachi, Pakistan for the next **72 hours**. Built with real atmospheric data, automated retraining, and a live interactive dashboard.

---

## 📌 Live Demo

> Run locally with `streamlit run dashboard/app.py`

---

## 🚀 Key Features

- **Real-time data** from Open-Meteo API + WAQI (US Embassy Karachi ground station)
- **72-hour recursive forecasting** using XGBoost with 44 engineered features
- **EPA 2024 AQI breakpoints** — updated May 6, 2024 standards
- **SHAP explainability** — every prediction explained, no black box
- **Automated CI/CD** — GitHub Actions retrains model daily
- **3-page interactive dashboard** — Dashboard, Forecast, Analysis

---

## 📊 Model Performance

| Model | MAE (µg/m³) | RMSE | R² Score |
|---|---|---|---|
| Ridge Regression | 0.853 | 1.332 | 0.980 |
| Random Forest | 0.937 | 1.622 | 0.970 |
| **XGBoost ✓** | **0.779** | **1.378** | **0.978** |

**Multi-step forecast accuracy:**

| Horizon | MAE (µg/m³) |
|---|---|
| 24 hours | 7.5 |
| 48 hours | 8.4 |
| 72 hours | 8.8 |

> MAE increases with forecast horizon — expected behavior showing the model is not overfit.

---

## 🏗️ Architecture

```
Raw Data (Open-Meteo + WAQI)
        ↓
  fetch_data.py          ← Pulls hourly weather + pollution data
        ↓
feature_engineering.py   ← 44 features: lag, rolling, cyclic, interaction
        ↓
   train_model.py        ← Trains Ridge, Random Forest, XGBoost — picks best
        ↓
  forecast_model.py      ← Recursive 72-hour multi-step prediction
        ↓
  explain_model.py       ← SHAP feature importance
        ↓
   dashboard/app.py      ← Streamlit 3-page interactive UI
        ↓
  GitHub Actions         ← Daily auto-retrain at 00:00 UTC
```

---

## ⚙️ Feature Engineering (44 Features)

| Category | Features | Purpose |
|---|---|---|
| **Lag Features** | pm2_5_lag1, lag3, lag6, lag12, lag24, lag48 | Temporal autocorrelation |
| **Rolling Averages** | pm2_5_roll3, roll6, roll12, roll24 + AQI versions | Short & long term trends |
| **Cyclic Encoding** | hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos | Periodic time patterns |
| **Time Features** | hour, day_of_week, month, is_weekend, is_rush_hour | Daily/weekly patterns |
| **Weather** | temperature, humidity, wind_speed, wind_dir, pressure | Atmospheric dispersion |
| **Pollutants** | pm10, no2, ozone, co, so2 | Multi-pollutant correlation |
| **Interaction** | pm_ratio, wind_dispersion, heat_humidity | Non-linear relationships |

---

## 🔍 Top SHAP Features

| Feature | SHAP Score | Why It Matters |
|---|---|---|
| 3hr Rolling Avg | 7.09 | PM2.5 has strong momentum — stays high if high |
| PM10 | 1.96 | Shares sources with PM2.5 — traffic, dust |
| PM Ratio | 1.19 | PM2.5/PM10 ratio identifies pollution type |
| Wind Dispersion | 0.77 | Strong wind physically disperses pollution |
| Rush Hour | 0.32 | Karachi traffic peaks at 8am and 6pm |

---

## 🗂️ Project Structure

```
karachi-aqi-predictor/
├── src/
│   ├── fetch_data.py          # Data ingestion — Open-Meteo + WAQI APIs
│   ├── feature_engineering.py # 44 feature pipeline + EPA 2024 AQI formula
│   ├── train_model.py         # Model training + comparison + serialization
│   ├── forecast_model.py      # Recursive 72-hour multi-step forecasting
│   └── explain_model.py       # SHAP explainability + visualization
├── dashboard/
│   └── app.py                 # Streamlit 3-page interactive dashboard
├── models/                    # Saved models + SHAP plots + results JSON
├── data/                      # CSV data files (gitignored)
├── .github/
│   └── workflows/
│       └── retrain.yml        # GitHub Actions daily retraining pipeline
├── requirements.txt
└── README.md
```

---

## 🚦 AQI Scale (EPA 2024 Standards)

| AQI | Category | PM2.5 (µg/m³) | Health Implications |
|---|---|---|---|
| 0–50 | 🟢 Good | 0–9.0 | Safe for all |
| 51–100 | 🟡 Moderate | 9.1–35.4 | Sensitive groups take caution |
| 101–150 | 🟠 Unhealthy (Sensitive) | 35.5–55.4 | Limit outdoor exposure |
| 151–200 | 🔴 Unhealthy | 55.5–125.4 | Wear mask outdoors |
| 201–300 | 🟣 Very Unhealthy | 125.5–225.4 | Avoid outdoor activity |
| 301+ | 🔴 Hazardous | 225.5+ | Stay indoors |

> Breakpoints updated to **EPA May 2024 standards** — stricter than old 2012 values.

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/MufeedHaider/karachi-aqi-predictor.git
cd karachi-aqi-predictor
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add WAQI token (optional — for real ground sensor data)
Create a `.env` file:
```
WAQI_TOKEN=your_token_here
```
Get your free token at: https://aqicn.org/data-platform/token/

### 5. Run the full pipeline
```bash
python src/fetch_data.py
python src/feature_engineering.py
python src/train_model.py
python src/forecast_model.py
python src/explain_model.py
```

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔄 Data Sources

| Source | Type | Usage |
|---|---|---|
| **Open-Meteo Air Quality API** | Atmospheric model | 1-year historical training data |
| **Open-Meteo Archive API** | Weather model | Historical temperature, wind, humidity |
| **Open-Meteo Forecast API** | 3-day forecast | Future weather for forecast features |
| **WAQI (aqicn.org)** | Ground sensor | Real-time AQI from US Embassy Karachi |

---

## 🤖 MLOps Pipeline

- **Automated ingestion** — hourly data fetch via GitHub Actions
- **Daily retraining** — model updates every 24 hours at midnight UTC
- **Feature drift handling** — rolling window ensures model stays current
- **Model versioning** — best model serialized with pickle, results logged to JSON
- **Explainability** — SHAP TreeExplainer runs after every retrain

---

## 📈 Dashboard Pages

### 1. Dashboard
- Live AQI card with color-coded status
- Weather strip (temperature, humidity, wind, pressure, PM2.5, PM10, CO)
- 72-hour AQI + PM2.5 dual-axis forecast chart
- 7-day historical AQI trend
- Next 24 hours hourly table
- Pollutant breakdown with WHO limits
- SHAP feature importance chart

### 2. Forecast
- Model accuracy metrics (24hr, 48hr, 72hr MAE)
- Full 72-hour chart with AQI category coloring
- Complete 72-row hourly breakdown table

### 3. Analysis
- Model comparison table (Ridge vs Random Forest vs XGBoost)
- Full SHAP importance chart
- Feature engineering breakdown
- Top feature explanations in plain English

---

## 🧪 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Models | XGBoost, Random Forest, Ridge Regression |
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Streamlit + Plotly |
| Data APIs | Open-Meteo, WAQI |
| Automation | GitHub Actions |
| Data Processing | Pandas, NumPy, Scikit-learn |

---

## 👤 Author

**Mufeed Haider**
- GitHub: [@MufeedHaider](https://github.com/MufeedHaider)
- Email: mufeedzaidi786@gmail.com

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built as part of the 10Pearls Shine Internship Program — Data Science & AI Track*

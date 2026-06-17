🌫️ Karachi AQI Predictor

An end-to-end MLOps pipeline for forecasting PM2.5 air quality in Karachi, Pakistan — 24 to 72 hours ahead — with automated daily retraining, explainable AI, and a live interactive dashboard.


📊 Model Performance

ModelMAE (µg/m³)RMSER²Ridge Regression0.8531.3320.980Random Forest0.9371.6220.970XGBoost ✅0.7791.3780.978

Forecast HorizonMAE (µg/m³)24-hour ahead7.5448-hour ahead8.4472-hour ahead8.76


AQI breakpoints follow EPA 2024 updated standards (effective May 6, 2024).




🏗️ Architecture

Open-Meteo API (1yr historical)          WAQI API (live ground sensor)
        │                                         │
        ▼                                         ▼
  fetch_data.py  ──────────────────────────────────
        │
        ▼
feature_engineering.py
  • 44 engineered features
  • Lag features (1h, 3h, 6h, 12h, 24h, 48h)
  • Rolling averages (3h, 6h, 12h, 24h)
  • Cyclic encoding (hour, day, month)
  • Weather interaction features
  • EPA 2024 multi-pollutant AQI
        │
        ▼
  train_model.py
  • Ridge / Random Forest / XGBoost comparison
  • Time-series split (no data leakage)
  • Best model saved as best_model.pkl
        │
        ▼
  explain_model.py          forecast_model.py
  • SHAP TreeExplainer       • Recursive multi-step
  • Feature importance         forecasting
  • Beeswarm plots             24hr / 48hr / 72hr
        │                           │
        └──────────┬────────────────┘
                   ▼
           dashboard/app.py
           • Streamlit 3-page UI
           • Live AQI from US Embassy station
           • 72hr forecast chart
           • SHAP explainability view
                   │
                   ▼
         GitHub Actions (daily)
         • Fetches fresh data
         • Retrains model
         • Updates forecast


📁 Project Structure

karachi-aqi-predictor/
├── src/
│   ├── fetch_data.py          # Data ingestion (Open-Meteo + WAQI)
│   ├── feature_engineering.py # 44 features, EPA 2024 AQI
│   ├── train_model.py         # Model training + evaluation
│   ├── explain_model.py       # SHAP explainability
│   └── forecast_model.py      # Recursive 72hr forecasting
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── models/
│   ├── best_model.pkl         # Trained XGBoost model
│   ├── shap_importance.png    # Feature importance chart
│   └── all_results.json       # Model comparison results
├── data/                      # Auto-generated (gitignored)
├── .github/workflows/
│   └── retrain.yml            # Daily retraining pipeline
├── requirements.txt
└── README.md


🔬 Features Engineered (44 total)

Pollutants: PM2.5, PM10, NO2, Ozone, CO, SO2

Weather: Temperature, Humidity, Wind Speed, Wind Direction, Pressure

Temporal: Hour, Day of Week, Month, Is Weekend, Is Rush Hour

Cyclic Encoding: sin/cos of hour, day-of-week, month (prevents discontinuity at midnight/week boundaries)

Lag Features: PM2.5 and AQI at 1h, 3h, 6h, 12h, 24h, 48h prior

Rolling Averages: PM2.5 and AQI over 3h, 6h, 12h, 24h windows

Interaction Features:


pm_ratio = PM2.5 / PM10 — indicates combustion vs dust pollution
wind_dispersion = wind speed / PM2.5 — atmospheric dispersion proxy
heat_humidity = temperature × humidity — thermal pollution trapping



🧠 Top SHAP Features

| Feature | Mean |SHAP| | Interpretation |
|---|---|---|
| pm2_5_roll3 | 7.09 | 3-hour momentum is the strongest predictor |
| pm10 | 1.96 | Coarse particle correlation with PM2.5 |
| pm_ratio | 1.19 | Pollution source type indicator |
| wind_dispersion | 0.77 | Wind clearing effect |
| pm2_5_lag3 | 0.65 | Recent history |
| is_rush_hour | 0.32 | Karachi traffic peak (8am, 6pm) |


🚀 Setup & Run

bash# 1. Clone
git clone https://github.com/MufeedHaider/karachi-aqi-predictor.git
cd karachi-aqi-predictor

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add WAQI token (free at aqicn.org/data-platform/token)
echo WAQI_TOKEN=your_token_here > .env

# 5. Run full pipeline
python src/fetch_data.py
python src/feature_engineering.py
python src/train_model.py
python src/explain_model.py
python src/forecast_model.py

# 6. Launch dashboard
streamlit run dashboard/app.py


📡 Data Sources

SourcePurposeCostOpen-Meteo Air Quality API1-year hourly historical training dataFree, no keyOpen-Meteo Archive APIHistorical weather dataFree, no keyWAQI APILive reading from US Embassy Karachi ground stationFree token

Why hybrid? Open-Meteo provides 8,760 rows of hourly historical data needed for model training. WAQI provides real ground-sensor accuracy for the live dashboard reading — matching IQAir quality.


⚙️ MLOps — Automated Daily Retraining

A GitHub Actions workflow runs every day at midnight PKT:


Fetches the latest 24 hours of data
Appends to training dataset
Retrains XGBoost model
Updates the 72-hour forecast
Commits updated model artifacts back to the repo



🎯 Interview Talking Points


Why XGBoost over Random Forest? Boosting corrects previous tree errors sequentially. In time-series data with autocorrelation, this outperforms bagging approaches like RF.
Why recursive forecasting? Target-shifting (predicting 72h ahead directly) degrades because features become stale. Recursive forecasting feeds each prediction as input to the next step, maintaining feature consistency.
Why cyclic encoding? Raw hour values (23 → 0) create a false discontinuity. sin/cos encoding preserves the circular nature of time.
Data source limitation: Open-Meteo is an atmospheric model, not ground sensors. For a production system, integrating Pakistan EPA sensor networks would improve real-time accuracy. Our value is the 72-hour forecast trend.
EPA 2024 update: We use the updated May 2024 breakpoints where "Good" threshold lowered from 12 to 9 µg/m³, reflecting stricter WHO alignment.



👤 Author

Mufeed Haider
Data Science | ML Engineering
Karachi, Pakistan


Data refreshed daily. AQI calculated per US EPA 2024 standards.
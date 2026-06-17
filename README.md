# 🌫️ Karachi AQI Predictor

An end-to-end Machine Learning and MLOps project for forecasting **PM2.5 air pollution levels in Karachi, Pakistan**, up to **72 hours ahead**, using advanced feature engineering, explainable AI, and an interactive Streamlit dashboard.

The project combines historical atmospheric data, live ground-sensor readings, and machine learning to provide actionable air-quality forecasts and insights.

---

## 🎯 Project Highlights

* 📈 Forecast PM2.5 levels 24, 48, and 72 hours ahead
* 🤖 Compare Ridge Regression, Random Forest, and XGBoost
* 🔍 Explain predictions using SHAP (Explainable AI)
* 🌦️ Integrate weather and pollution signals
* 📊 Interactive Streamlit dashboard
* ⚙️ End-to-end reproducible ML pipeline
* 🚀 Production-oriented MLOps workflow

---

## 📊 Model Performance

### Model Comparison

| Model            | MAE (µg/m³) | RMSE      | R²        |
| ---------------- | ----------- | --------- | --------- |
| Ridge Regression | 0.853       | 1.332     | 0.980     |
| Random Forest    | 0.937       | 1.622     | 0.970     |
| **XGBoost** ✅    | **0.779**   | **1.378** | **0.978** |

### Forecast Horizon Performance

| Forecast Horizon | MAE (µg/m³) |
| ---------------- | ----------- |
| 24-hour Ahead    | 7.54        |
| 48-hour Ahead    | 8.44        |
| 72-hour Ahead    | 8.76        |

> AQI calculations follow the updated EPA 2024 standards (effective May 2024).

---

## 🏗️ System Architecture

```text
Open-Meteo Historical APIs             WAQI Live Sensor API
        │                                      │
        ▼                                      ▼
                 Data Collection
                       │
                       ▼
                 Feature Engineering
        • Lag Features
        • Rolling Statistics
        • Cyclic Encoding
        • Weather Interactions
        • EPA 2024 AQI Calculation
                       │
                       ▼
                 Model Training
        • Ridge Regression
        • Random Forest
        • XGBoost
                       │
                       ▼
              Explainability (SHAP)
                       │
                       ▼
                Forecast Generation
                 (24h / 48h / 72h)
                       │
                       ▼
                Streamlit Dashboard
                       │
                       ▼
               Automated Retraining
```

---

## 📁 Project Structure

```text
karachi-aqi-predictor/
│
├── src/
│   ├── fetch_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── explain_model.py
│   └── forecast_model.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── best_model.pkl
│   ├── all_results.json
│   ├── horizon_results.json
│   ├── shap_importance.png
│   └── shap_summary.png
│
├── data/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Feature Engineering

A total of **44 engineered features** were created to capture temporal patterns, weather effects, and pollution dynamics.

### Air Pollutants

* PM2.5
* PM10
* NO₂
* O₃ (Ozone)
* CO
* SO₂

### Weather Variables

* Temperature
* Relative Humidity
* Wind Speed
* Wind Direction
* Atmospheric Pressure

### Temporal Features

* Hour of Day
* Day of Week
* Month
* Weekend Indicator
* Rush Hour Indicator

### Cyclic Encoding

To preserve the cyclical nature of time:

* Hour → sin/cos
* Day of Week → sin/cos
* Month → sin/cos

### Lag Features

Historical PM2.5 and AQI values at:

* 1 hour
* 3 hours
* 6 hours
* 12 hours
* 24 hours
* 48 hours

### Rolling Statistics

Moving averages over:

* 3 hours
* 6 hours
* 12 hours
* 24 hours

### Interaction Features

**pm_ratio**

```text
PM2.5 / PM10
```

**wind_dispersion**

```text
Wind Speed / PM2.5
```

**heat_humidity**

```text
Temperature × Humidity
```

---

## 🧠 Model Explainability (SHAP)

The project uses SHAP (SHapley Additive exPlanations) to interpret model predictions and identify the most influential factors affecting PM2.5 levels.

### Top SHAP Features

| Feature         | Mean SHAP Value | Interpretation                             |
| --------------- | --------------- | ------------------------------------------ |
| pm2_5_roll3     | 7.09            | Strong short-term pollution momentum       |
| pm10            | 1.96            | Correlation with coarse particulate matter |
| pm_ratio        | 1.19            | Indicates pollution source characteristics |
| wind_dispersion | 0.77            | Wind-driven pollutant dispersion           |
| pm2_5_lag3      | 0.65            | Recent pollution history                   |
| is_rush_hour    | 0.32            | Traffic-related emissions                  |

---

## 🚀 Installation & Usage

### Clone Repository

```bash
git clone https://github.com/MufeedHaider/karachi-aqi-predictor.git
cd karachi-aqi-predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure WAQI Token

```bash
WAQI_TOKEN=your_token_here
```

### Run Full Pipeline

```bash
python src/fetch_data.py
python src/feature_engineering.py
python src/train_model.py
python src/explain_model.py
python src/forecast_model.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📡 Data Sources

| Source                     | Purpose                                                   |
| -------------------------- | --------------------------------------------------------- |
| Open-Meteo Air Quality API | Historical air-quality training data                      |
| Open-Meteo Archive API     | Historical weather data                                   |
| WAQI API                   | Live air-quality readings from Karachi monitoring station |

### Why a Hybrid Approach?

Open-Meteo provides large-scale historical data suitable for model training, while WAQI provides live ground-sensor observations used for real-time dashboard updates and forecast validation.

---

## ⚙️ MLOps Workflow

The automated workflow is designed to:

1. Collect fresh environmental data
2. Update the training dataset
3. Retrain the forecasting model
4. Generate updated forecasts
5. Refresh model artifacts and visualizations

This ensures that predictions remain current as environmental conditions evolve.

---

## 👨‍💻 Author

**Mufeed Haider**

Data Science • Machine Learning • MLOps

Karachi, Pakistan

GitHub: https://github.com/MufeedHaider

---

⭐ If you found this project interesting, consider giving the repository a star.

"""
SHAP explainability for the 24-hour forecast model.

Explaining the right model matters here. The previous version explained a model
whose target was the current hour and whose top three features were arithmetic
restatements of that target, so the explanation was of the leak rather than of
air quality: "the 3-hour mean predicts PM2.5" is true and empty when the 3-hour
mean contains PM2.5.

This runs on the 24-hour-ahead model, where every feature is genuinely observed
a day before the value being predicted, so the attributions describe something
real. Note that the model predicts a *change* in PM2.5, so SHAP values are in
µg/m³ of predicted change, not of absolute concentration.

Outputs models/shap_importance.csv, which the dashboard reads directly — the
dashboard no longer carries hardcoded copies of these numbers.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import build_supervised  # noqa: E402

warnings.filterwarnings("ignore")

EXPLAIN_HORIZON = 24
SAMPLE_SIZE = 500

# Human-readable names, used for both the plots and the dashboard.
PRETTY_NAMES = {
    "cams_pm25": "CAMS PM2.5 (now)",
    "cams_pm10": "CAMS PM10",
    "fut_cams_pm25": "CAMS forecast for target hour",
    "cams_ratio": "Measured / CAMS ratio (now)",
    "cams_ratio_roll24": "Measured / CAMS ratio (24h mean)",
    "cams_gap": "Measured minus CAMS (now)",
    "pm25_lag1": "Measured PM2.5 1h ago",
    "pm25_lag3": "Measured PM2.5 3h ago",
    "pm25_lag6": "Measured PM2.5 6h ago",
    "pm25_lag12": "Measured PM2.5 12h ago",
    "pm25_lag24": "Measured PM2.5 24h ago",
    "pm25_lag48": "Measured PM2.5 48h ago",
    "pm25_roll3": "Measured PM2.5 3h mean",
    "pm25_roll6": "Measured PM2.5 6h mean",
    "pm25_roll12": "Measured PM2.5 12h mean",
    "pm25_roll24": "Measured PM2.5 24h mean",
    "pm25_std24": "Measured PM2.5 24h volatility",
    "pm_ratio": "CAMS PM2.5 / PM10 ratio",
    "wind_dispersion": "Wind dispersion",
    "heat_humidity": "Heat x humidity",
    "fut_wind_speed": "Forecast wind speed",
    "fut_temperature": "Forecast temperature",
    "fut_humidity": "Forecast humidity",
    "fut_pressure": "Forecast pressure",
    "fut_wind_dir": "Forecast wind direction",
    "fut_hour_sin": "Target hour (sin)",
    "fut_hour_cos": "Target hour (cos)",
    "fut_is_rush_hour": "Target hour is rush hour",
    "is_rush_hour": "Rush hour",
    "is_weekend": "Weekend",
    "wind_speed": "Wind speed",
    "wind_dir": "Wind direction",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "no2": "NO2", "so2": "SO2", "co": "CO", "ozone": "Ozone",
}


def pretty(name):
    if name in PRETTY_NAMES:
        return PRETTY_NAMES[name]
    if name.startswith("aqi_lag"):
        return f"AQI {name.replace('aqi_lag', '')}h ago"
    if name.startswith("aqi_roll"):
        return f"AQI {name.replace('aqi_roll', '')}h mean"
    return name.replace("_", " ")


def load_horizon_model(horizon):
    path = "models/horizon_models.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run src/forecast_model.py before src/explain_model.py"
        )
    with open(path, "rb") as f:
        models = pickle.load(f)
    return models[horizon]


def explain():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/featured_data.csv", parse_dates=["timestamp"])
    X, _, _, _ = build_supervised(df, EXPLAIN_HORIZON)
    model = load_horizon_model(EXPLAIN_HORIZON)

    sample = X.sample(min(SAMPLE_SIZE, len(X)), random_state=42)
    display = sample.rename(columns={c: pretty(c) for c in sample.columns})

    print(f"Computing SHAP values for the {EXPLAIN_HORIZON}-hour model "
          f"({len(sample)} sampled rows)...")

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, display, plot_type="bar", show=False, max_display=15)
    plt.title(
        f"Feature importance - {EXPLAIN_HORIZON}h-ahead measured PM2.5", fontsize=13
    )
    plt.xlabel("Mean |SHAP value| (µg/m³ of predicted change in measured PM2.5)")
    plt.tight_layout()
    plt.savefig("models/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, display, show=False, max_display=15)
    plt.title(
        f"SHAP summary - {EXPLAIN_HORIZON}h-ahead PM2.5 change", fontsize=13
    )
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = (
        pd.DataFrame(
            {
                "feature": sample.columns,
                "label": [pretty(c) for c in sample.columns],
                "mean_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["horizon_hours"] = EXPLAIN_HORIZON
    importance.to_csv("models/shap_importance.csv", index=False)

    print("\nTop 10 drivers of the 24-hour forecast:")
    print(importance.head(10)[["label", "mean_shap"]].to_string(index=False))
    print("\nSaved shap_importance.csv, shap_importance.png, shap_summary.png")


if __name__ == "__main__":
    explain()

"""
Data ingestion for the Karachi PM2.5 forecaster.

Writes three files, all from Open-Meteo:

  data/raw_air_quality.csv    — a year of hourly CAMS pollution + weather
  data/forecast_weather.csv   — forecast weather for the coming days
  data/reference_forecast.csv — the CAMS PM2.5 forecast

All three are model *inputs*, not the thing being predicted. The target is
measured PM2.5 from ground monitors.

Measured ground-sensor data is fetched separately by src/ground_truth.py, which
talks to OpenAQ. The WAQI feed this project originally used (the US Consulate
station) stopped reporting in March 2025 and has been removed rather than left
in place returning an eighteen-month-old value.

Credentials
-----------
API keys are read from the environment only. An earlier version carried a
working token as a hardcoded default, which meant it was published in a public
repository and in every commit of its history. Keys belong in GitHub Actions
secrets and a local .env file, never in source.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import requests

try:  # optional convenience for local runs
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"
TIMEOUT = 30

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

POLLUTANT_VARS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
    "sulphur_dioxide",
]
POLLUTANT_RENAME = {
    "nitrogen_dioxide": "no2",
    "carbon_monoxide": "co",
    "sulphur_dioxide": "so2",
}
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]
WEATHER_RENAME = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
    "surface_pressure": "pressure",
}


def _hourly_frame(url, params, variables, rename):
    """Call an Open-Meteo endpoint and return its hourly block as a DataFrame."""
    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError(f"no hourly block in response from {url}: {payload}")

    hourly = payload["hourly"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for var in variables:
        frame[rename.get(var, var)] = hourly[var]
    return frame


def fetch_history(start_date, end_date):
    """One year of hourly pollution and weather observations."""
    air = _hourly_frame(
        AIR_QUALITY_URL,
        {
            "latitude": LAT,
            "longitude": LON,
            "hourly": POLLUTANT_VARS,
            "timezone": TIMEZONE,
            "start_date": start_date,
            "end_date": end_date,
        },
        POLLUTANT_VARS,
        POLLUTANT_RENAME,
    )
    weather = _hourly_frame(
        ARCHIVE_URL,
        {
            "latitude": LAT,
            "longitude": LON,
            "hourly": WEATHER_VARS,
            "timezone": TIMEZONE,
            "start_date": start_date,
            "end_date": end_date,
        },
        WEATHER_VARS,
        WEATHER_RENAME,
    )
    merged = pd.merge(air, weather, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def fetch_forecast_weather(days=4):
    """Forecast weather for the coming days.

    This is a real model input: when forecasting PM2.5 at t+48, the weather
    expected at t+48 is information that genuinely exists at t.
    """
    return _hourly_frame(
        FORECAST_URL,
        {
            "latitude": LAT,
            "longitude": LON,
            "hourly": WEATHER_VARS,
            "timezone": TIMEZONE,
            "forecast_days": days,
        },
        WEATHER_VARS,
        WEATHER_RENAME,
    )


def fetch_reference_forecast(days=4):
    """The CAMS PM2.5 forecast, used both as a feature and as the benchmark.

    This is legitimate here precisely because the target is a *measurement*, not
    CAMS itself. The model is told what the physics simulation expects for the
    target hour and, separately, how far that simulation currently sits from the
    monitors — so it can use the transport and dust signal while correcting the
    bias. Over Karachi that bias is large: CAMS reads about 37% low on average
    and 2.4x low in December.

    The same series is the baseline the model is scored against. Beating it is
    the test of whether this project adds anything to the physics.
    """
    return _hourly_frame(
        AIR_QUALITY_URL,
        {
            "latitude": LAT,
            "longitude": LON,
            "hourly": POLLUTANT_VARS,
            "timezone": TIMEZONE,
            "forecast_days": days,
        },
        POLLUTANT_VARS,
        POLLUTANT_RENAME,
    )


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    today = datetime.today()
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"[1/3] Historical observations {start_date} -> {end_date}")
    history = fetch_history(start_date, end_date)
    history["is_forecast"] = False
    history.to_csv("data/raw_air_quality.csv", index=False)
    print(f"  {len(history)} hourly rows -> data/raw_air_quality.csv")

    print("\n[2/3] Forecast weather (model input)")
    try:
        weather = fetch_forecast_weather()
        weather.to_csv("data/forecast_weather.csv", index=False)
        print(f"  {len(weather)} hourly rows -> data/forecast_weather.csv")
    except Exception as exc:
        print(f"  failed: {exc}")
        print("  the forecaster will carry the last observed weather forward")

    print("\n[3/3] CAMS PM2.5 forecast (model input and benchmark)")
    try:
        reference = fetch_reference_forecast()
        reference.to_csv("data/reference_forecast.csv", index=False)
        print(f"  {len(reference)} hourly rows -> data/reference_forecast.csv")
    except Exception as exc:
        print(f"  failed: {exc}")

    print("\nOpen-Meteo ingestion complete.")
    print("Measured ground data comes from src/ground_truth.py (OpenAQ).")


if __name__ == "__main__":
    main()

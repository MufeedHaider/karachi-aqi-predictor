"""
Feature engineering for the Karachi PM2.5 forecaster.

What is being predicted
-----------------------
The target is **measured** PM2.5 — the median of Karachi's live ground monitors
for a given hour, from OpenAQ. It is not CAMS model output.

That distinction is the point of this project. CAMS (the Copernicus atmospheric
model, via Open-Meteo) is a 45 km global simulation, and over Karachi it
under-reports badly: across a year of overlap it read 37% low on average, and
2.4x low in December, missing the winter pollution season almost entirely. A
model trained to predict CAMS would inherit that error and call it accuracy.

So CAMS is demoted from target to *feature*. It carries real physical
information — transport, chemistry, regional dust — and the model learns how far
from reality it usually sits, given the season, the hour and the weather.

On the forecasting target
-------------------------
Every feature is observed at time *t*; the label is measured PM2.5 at
*t + horizon*. An earlier version of this project trained on `pm2_5[t]` while
feeding it features computed from `pm2_5[t]`, which made the target
algebraically recoverable from the inputs and produced a meaningless R² of
0.978. `build_supervised` makes that impossible by construction.

Models predict the *change* from the current reading rather than the level.
Gradient-boosted trees cannot extrapolate a near-identity mapping, so asking for
the level makes them lose to a naive "nothing changes" baseline at short leads.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from aqi import aqi_from_pm25, overall_aqi

# Outlier caps and any other fitted quantity must come from training rows only.
TRAIN_FRACTION = 0.8

CAMS_POLLUTANTS = ["pm2_5", "pm10", "no2", "ozone", "co", "so2"]
WEATHER_COLS = ["temperature", "humidity", "wind_speed", "wind_dir", "pressure"]

# Ground monitors are low-cost optical sensors. Readings outside this range are
# hardware faults, not air quality.
GROUND_MIN, GROUND_MAX = 0.5, 900.0

BASE_FEATURES = [
    # CAMS simulation at time t — a physics prior, not the answer
    "cams_pm25", "cams_pm10", "no2", "ozone", "co", "so2",
    # observed weather
    "temperature", "humidity", "wind_speed", "wind_dir", "pressure",
    # calendar
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    # measured PM2.5 history
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12", "pm25_lag24", "pm25_lag48",
    "pm25_roll3", "pm25_roll6", "pm25_roll12", "pm25_roll24", "pm25_std24",
    # how wrong CAMS is *right now* — the model's handle on the bias
    "cams_ratio", "cams_ratio_roll24", "cams_gap",
    # interactions
    "pm_ratio", "wind_dispersion", "heat_humidity",
]

# Known at forecast time: tomorrow's weather forecast and tomorrow's CAMS run
# both exist today.
FUTURE_FEATURES = [
    "fut_temperature", "fut_humidity", "fut_wind_speed", "fut_wind_dir",
    "fut_pressure", "fut_cams_pm25",
    "fut_hour_sin", "fut_hour_cos", "fut_is_rush_hour",
]

MODEL_FEATURES = BASE_FEATURES + FUTURE_FEATURES

# Columns that must never become features: they are the answer, or derived from it.
BANNED_AS_FEATURES = {"ground_pm25", "aqi", "timestamp", "n_stations", "is_forecast"}


def load_sources(cams_path="data/raw_air_quality.csv",
                 ground_path="data/ground_history.csv"):
    """Merge the CAMS archive with measured ground-sensor history."""
    if not os.path.exists(ground_path):
        raise SystemExit(
            f"{ground_path} not found.\n"
            "The model trains on measured PM2.5, so this file is required.\n"
            "Run:  python src/ground_truth.py --history 400"
        )

    cams = pd.read_csv(cams_path, parse_dates=["timestamp"])
    ground = pd.read_csv(ground_path, parse_dates=["timestamp"])

    if "is_forecast" in cams.columns:
        cams = cams[~cams["is_forecast"].astype(bool)]

    ground = ground.rename(columns={"pm2_5": "ground_pm25"})
    ground = ground[
        ground["ground_pm25"].between(GROUND_MIN, GROUND_MAX)
    ]

    merged = pd.merge(
        cams,
        ground[["timestamp", "ground_pm25", "n_stations"]],
        on="timestamp",
        how="inner",
    )
    return merged.sort_values("timestamp").reset_index(drop=True)


def _fill_and_cap(df, cols):
    """Forward-fill only, then cap on training-period percentiles.

    Interpolation would average the previous and *next* observation, pulling
    future values of the target into its own history.
    """
    n_train = int(len(df) * TRAIN_FRACTION)
    for col in cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].iloc[:n_train].median())
            train_slice = df[col].iloc[:n_train]
            df[col] = df[col].clip(
                train_slice.quantile(0.005), train_slice.quantile(0.995)
            )
    return df


def engineer_features(df, out_path="data/featured_data.csv"):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Put the series on a strict hourly grid so lags mean what they say. A gap
    # of two hours must not let lag1 reach back three.
    df = df.set_index("timestamp").asfreq("h").reset_index()

    df = _fill_and_cap(df, CAMS_POLLUTANTS + WEATHER_COLS)
    df["ground_pm25"] = df["ground_pm25"].ffill(limit=3)

    df = df.rename(columns={"pm2_5": "cams_pm25", "pm10": "cams_pm10"})

    # AQI of the measured value — this is what the dashboard reports.
    df["aqi"] = df["ground_pm25"].map(aqi_from_pm25)
    df["cams_aqi"] = [
        overall_aqi(pm25=r.cams_pm25, pm10=r.cams_pm10, no2_ugm3=r.no2)
        for r in df.itertuples()
    ]

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"pm25_lag{lag}"] = df["ground_pm25"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"pm25_roll{window}"] = df["ground_pm25"].rolling(window).mean()
    df["pm25_std24"] = df["ground_pm25"].rolling(24).std()

    # The bias signal. How far the simulation sits from the measurement right
    # now is the single most useful thing the model can know about CAMS.
    df["cams_ratio"] = df["ground_pm25"] / (df["cams_pm25"] + 1e-6)
    df["cams_ratio_roll24"] = df["cams_ratio"].rolling(24).mean()
    df["cams_gap"] = df["ground_pm25"] - df["cams_pm25"]

    df["pm_ratio"] = df["cams_pm25"] / (df["cams_pm10"] + 1e-6)
    df["wind_dispersion"] = df["wind_speed"] / (df["ground_pm25"] + 1e-6)
    df["heat_humidity"] = df["temperature"] * df["humidity"] / 100

    before = len(df)
    df = df.dropna(subset=BASE_FEATURES + ["ground_pm25"]).reset_index(drop=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)

    print(f"Feature table: {len(df)} rows x {df.shape[1]} columns")
    print(f"  dropped {before - len(df)} rows (lag warm-up or missing measurements)")
    print(f"  model features: {len(MODEL_FEATURES)}")
    print(f"  period: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  measured PM2.5  mean {df['ground_pm25'].mean():.1f}  "
          f"median {df['ground_pm25'].median():.1f}  max {df['ground_pm25'].max():.1f}")
    print(f"  CAMS PM2.5      mean {df['cams_pm25'].mean():.1f}  "
          f"median {df['cams_pm25'].median():.1f}")
    gap = df["ground_pm25"].mean() - df["cams_pm25"].mean()
    direction = "below" if gap > 0 else "above"
    print(f"  CAMS reads {abs(gap):.1f} ug/m3 "
          f"({100 * abs(gap) / df['ground_pm25'].mean():.0f}%) {direction} measurement")
    print(f"  stations per hour: median {df['n_stations'].median():.0f}")
    return df


def build_supervised(df, horizon):
    """Assemble (X, y_delta, current, target_timestamp) for one lead time.

    X holds only quantities knowable at time t. y_delta is the change in
    measured PM2.5 between t and t + horizon.
    """
    out = df[BASE_FEATURES].copy()

    for col in WEATHER_COLS:
        out["fut_" + col] = df[col].shift(-horizon)
    out["fut_cams_pm25"] = df["cams_pm25"].shift(-horizon)

    target_ts = df["timestamp"].shift(-horizon)
    fut_hour = target_ts.dt.hour
    out["fut_hour_sin"] = np.sin(2 * np.pi * fut_hour / 24)
    out["fut_hour_cos"] = np.cos(2 * np.pi * fut_hour / 24)
    out["fut_is_rush_hour"] = fut_hour.isin([7, 8, 9, 17, 18, 19]).astype(float)

    current = df["ground_pm25"]
    future = df["ground_pm25"].shift(-horizon)
    y_delta = future - current

    # Only score against hours that are genuinely `horizon` apart. After
    # reindexing to an hourly grid a shift is a true clock offset, but rows
    # whose target hour was dropped must still be excluded.
    exact = (target_ts - df["timestamp"]) == pd.Timedelta(hours=horizon)
    valid = out.notna().all(axis=1) & y_delta.notna() & exact

    return (
        out.loc[valid, MODEL_FEATURES],
        y_delta[valid],
        current[valid],
        target_ts[valid],
    )


def train_test_split_index(n):
    """Chronological split point. Never shuffle a time series."""
    return int(n * TRAIN_FRACTION)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    engineer_features(load_sources())

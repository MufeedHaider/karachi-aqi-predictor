"""
72-hour forecast of measured PM2.5: one direct model per lead time.

Direct, not recursive
---------------------
An earlier version predicted t+1, fed that prediction back as if it were an
observation, and repeated 72 times. Features changed meaning between fitting and
serving, every exogenous input was frozen at its seed value, and the published
forecast came out as a flat line varying by 2 µg/m³ over three days.

Seventy-two separate models remove the feedback entirely. Model *h* sees only
what is observable at time *t* and predicts directly at *t+h*.

What each model gets for the target hour
----------------------------------------
Two things that genuinely exist at forecast time:

  * the Open-Meteo **weather forecast** for that hour
  * the **CAMS PM2.5 forecast** for that hour

The second is the interesting one. CAMS is a real operational product, and over
Karachi it reads about 37% low — but it is not noise: it carries the regional
transport and dust signal. Handing it to the model as a feature, alongside the
measured history that reveals how far off it currently is, lets the model use
the physics and correct the bias at the same time. The backtest says that
combination beats CAMS alone by 30-40%.

Backtesting caveat, stated plainly
----------------------------------
Backtests substitute *analysed* weather and CAMS for *forecast* weather and
CAMS, because neither archive stores what was predicted at the time. Real
forecasts carry their own error, so live accuracy will be somewhat below the
backtested figures. The CAMS baseline is handicapped identically, so the
comparison between them stays fair.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aqi import aqi_from_pm25, category  # noqa: E402
from evaluation import rolling_origin  # noqa: E402
from feature_engineering import (  # noqa: E402
    MODEL_FEATURES,
    WEATHER_COLS,
    build_supervised,
    train_test_split_index,
)

MAX_HORIZON = 72
REPORT_HORIZONS = [1, 3, 6, 12, 24, 48, 72]


def make_model():
    """Small on purpose: within 1% of a model three times the size, and 72 of
    them have to fit inside a CI run."""
    return xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


def fit_all_horizons(df, backtest_horizons=REPORT_HORIZONS):
    """Fit the production model for every horizon; backtest a representative set.

    Production models train on all available data. The reported metrics come
    from rolling-origin folds, so they describe genuinely unseen periods rather
    than the data the shipped model was fitted on.
    """
    models, metrics = {}, {}

    for h in range(1, MAX_HORIZON + 1):
        X, y_delta, current, target_ts = build_supervised(df, h)

        if h in backtest_horizons:
            def fit_predict(X_tr, d_tr, X_te):
                model = make_model()
                model.fit(X_tr, d_tr)
                return model.predict(X_te)

            per_fold, combined = rolling_origin(
                X, y_delta, current, target_ts,
                fit_predict, cams=X["fut_cams_pm25"],
            )
            combined["horizon_hours"] = h
            combined["per_fold"] = per_fold
            metrics[h] = combined

        production = make_model()
        production.fit(X, y_delta)
        models[h] = production

    return models, metrics


def _aligned_future(path, anchor, columns):
    """Load a forecast file and index it by hours ahead of `anchor`."""
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as exc:  # pragma: no cover
        print(f"  could not read {path}: {exc}")
        return None

    frame = frame[frame["timestamp"] > anchor].copy()
    if frame.empty:
        return None
    frame["horizon"] = (
        (frame["timestamp"] - anchor).dt.total_seconds() // 3600
    ).astype(int)
    frame = frame[frame["horizon"].between(1, MAX_HORIZON)]
    available = [c for c in columns if c in frame.columns]
    return frame.set_index("horizon")[available] if not frame.empty else None


def forecast_from_latest(models, df):
    """Produce the live 1-72 hour forecast from the most recent measured hour."""
    latest = df.iloc[-1]
    anchor = pd.to_datetime(latest["timestamp"])
    current = float(latest["ground_pm25"])

    weather = _aligned_future("data/forecast_weather.csv", anchor, WEATHER_COLS)
    cams = _aligned_future("data/reference_forecast.csv", anchor, ["pm2_5"])

    degraded = []
    if weather is None:
        degraded.append("weather")
    if cams is None:
        degraded.append("CAMS")

    base = {c: latest[c] for c in MODEL_FEATURES if not c.startswith("fut_")}
    rows = []

    for h in range(1, MAX_HORIZON + 1):
        target_ts = anchor + pd.Timedelta(hours=h)
        feat = dict(base)

        for col in WEATHER_COLS:
            value = None
            if weather is not None and h in weather.index:
                value = weather.loc[h, col] if col in weather.columns else None
            feat["fut_" + col] = (
                float(value) if value is not None and pd.notna(value)
                else float(latest[col])
            )

        if cams is not None and h in cams.index and pd.notna(cams.loc[h, "pm2_5"]):
            feat["fut_cams_pm25"] = float(cams.loc[h, "pm2_5"])
        else:
            feat["fut_cams_pm25"] = float(latest["cams_pm25"])

        feat["fut_hour_sin"] = np.sin(2 * np.pi * target_ts.hour / 24)
        feat["fut_hour_cos"] = np.cos(2 * np.pi * target_ts.hour / 24)
        feat["fut_is_rush_hour"] = float(target_ts.hour in (7, 8, 9, 17, 18, 19))

        X = pd.DataFrame([feat])[MODEL_FEATURES]
        pm25 = max(0.0, current + float(models[h].predict(X)[0]))
        aqi = aqi_from_pm25(pm25)

        rows.append(
            {
                "timestamp": target_ts,
                "horizon_hour": h,
                "pm2_5_predicted": round(pm25, 2),
                "aqi_predicted": aqi,
                "aqi_category": category(aqi),
                "cams_pm2_5": round(feat["fut_cams_pm25"], 2),
            }
        )

    return pd.DataFrame(rows), degraded


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv("data/featured_data.csv", parse_dates=["timestamp"])
    print(f"Dataset: {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Target: measured PM2.5 (mean {df['ground_pm25'].mean():.1f} ug/m3)\n")

    print(f"Fitting {MAX_HORIZON} direct models and backtesting "
          f"{len(REPORT_HORIZONS)} of them...")
    models, metrics = fit_all_horizons(df)

    print("\nRolling-origin backtest (every season tested):")
    print(f"  {'lead':>6}{'MAE':>8}{'persist':>9}{'CAMS':>8}"
          f"{'vs persist':>12}{'vs CAMS':>10}{'R2':>8}")
    for h in REPORT_HORIZONS:
        m = metrics[h]
        persist = m["MAE"] / (1 - m["skill_vs_persistence"])
        print(
            f"  {str(h) + 'h':>6}{m['MAE']:>8.2f}{persist:>9.2f}{m['cams_MAE']:>8.2f}"
            f"{m['skill_vs_persistence']:>11.1%}{m['skill_vs_cams']:>10.1%}{m['R2']:>8.3f}"
        )

    mean_cams = float(np.mean([metrics[h]["skill_vs_cams"] for h in metrics]))
    mean_persist = float(np.mean([metrics[h]["skill_vs_persistence"] for h in metrics]))
    print(f"\n  Mean skill vs persistence: {mean_persist:.1%}")
    print(f"  Mean skill vs CAMS       : {mean_cams:.1%}")

    forecast_df, degraded = forecast_from_latest(models, df)
    forecast_df.to_csv("data/forecast_72hr.csv", index=False)

    if degraded:
        print(f"\n  WARNING: no live {' or '.join(degraded)} forecast available; "
              "last observed values carried forward.")

    print(f"\nForecast issued from {df['timestamp'].iloc[-1]} "
          f"(measured PM2.5 {df['ground_pm25'].iloc[-1]:.1f})")
    print(
        forecast_df.iloc[[0, 11, 23, 47, 71]][
            ["timestamp", "pm2_5_predicted", "aqi_predicted", "aqi_category"]
        ].to_string(index=False)
    )
    spread = forecast_df["pm2_5_predicted"]
    print(f"  range {spread.min():.1f} - {spread.max():.1f} ug/m3 (sd {spread.std():.2f})")

    recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=30)]
    recent[
        ["timestamp", "ground_pm25", "cams_pm25", "aqi", "n_stations",
         "temperature", "humidity", "wind_speed", "pressure"]
    ].to_csv("data/recent_history.csv", index=False)

    with open("models/horizon_results.json", "w") as f:
        json.dump(
            {
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "forecast_issued_at": str(df["timestamp"].iloc[-1]),
                "target": "measured PM2.5 (OpenAQ ground monitors, city median)",
                "evaluation": "rolling-origin backtest, 5 folds",
                "degraded_inputs": degraded,
                "mean_skill_vs_persistence": round(mean_persist, 3),
                "mean_skill_vs_cams": round(mean_cams, 3),
                "horizons": {
                    str(h): {k: v for k, v in metrics[h].items() if k != "per_fold"}
                    for h in sorted(metrics)
                },
                "folds": {
                    str(h): metrics[h]["per_fold"] for h in sorted(metrics)
                },
            },
            f,
            indent=2,
        )

    with open("models/horizon_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(MODEL_FEATURES, f)

    size_mb = os.path.getsize("models/horizon_models.pkl") / 1e6
    print(f"\nSaved {MAX_HORIZON} models ({size_mb:.1f} MB), metrics, forecast, history")


if __name__ == "__main__":
    main()

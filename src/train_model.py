"""
Algorithm selection for the measured-PM2.5 forecaster.

Compares Ridge, Random Forest and XGBoost against three baselines, at three
lead times, using rolling-origin backtesting so every season is tested. The
winner is written to models/best_model_name.txt and used by forecast_model.py.

The baseline that matters most is CAMS. It is not a strawman — it is the
Copernicus atmospheric model's own operational forecast for the same hour. If
this project cannot beat it, the project has no reason to exist.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import (  # noqa: E402
    climatology_map,
    metrics,
    print_fold_table,
    rolling_origin,
)
from feature_engineering import (  # noqa: E402
    BANNED_AS_FEATURES,
    MODEL_FEATURES,
    build_supervised,
)

SELECTION_HORIZON = 24
REPORT_HORIZONS = [1, 24, 72]


def candidates():
    return {
        "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=120, max_depth=14, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1,
        ),
    }


def assert_no_leakage():
    """The target is a future measurement; nothing derived from it may be input."""
    overlap = BANNED_AS_FEATURES.intersection(MODEL_FEATURES)
    if overlap:
        raise ValueError(
            f"Target-derived columns present in the feature list: {sorted(overlap)}. "
            "Features must be observable at time t."
        )


def evaluate_horizon(df, horizon):
    X, y_delta, current, target_ts = build_supervised(df, horizon)
    cams = X["fut_cams_pm25"]

    results = {}
    for name, factory in candidates().items():
        def fit_predict(X_tr, d_tr, X_te, _factory=factory):
            model = _factory()
            model.fit(X_tr, d_tr)
            return model.predict(X_te)

        per_fold, combined = rolling_origin(
            X, y_delta, current, target_ts, fit_predict, cams=cams
        )
        combined["per_fold"] = per_fold
        results[name] = combined

    # Baselines, scored on exactly the same pooled folds.
    def score_baseline(make_pred):
        def fit_predict(X_tr, d_tr, X_te):
            return make_pred(X_tr, d_tr, X_te)
        _, combined = rolling_origin(
            X, y_delta, current, target_ts, fit_predict, cams=cams
        )
        return combined

    results["Persistence (baseline)"] = score_baseline(
        lambda X_tr, d_tr, X_te: np.zeros(len(X_te))
    )

    hours = target_ts.dt.hour
    levels = current + y_delta

    def climatology(X_tr, d_tr, X_te):
        n_train = len(X_tr)
        mapping = climatology_map(hours.iloc[:n_train], levels.iloc[:n_train])
        overall = float(mapping.mean())
        target = hours.iloc[n_train:n_train + len(X_te)]
        predicted_level = target.map(mapping).fillna(overall).to_numpy()
        return predicted_level - current.iloc[n_train:n_train + len(X_te)].to_numpy()

    results["Hourly climatology (baseline)"] = score_baseline(climatology)

    def cams_raw(X_tr, d_tr, X_te):
        n_train = len(X_tr)
        return (
            X_te["fut_cams_pm25"].to_numpy()
            - current.iloc[n_train:n_train + len(X_te)].to_numpy()
        )

    results["CAMS forecast (baseline)"] = score_baseline(cams_raw)
    return results


def main():
    os.makedirs("models", exist_ok=True)
    assert_no_leakage()

    df = pd.read_csv("data/featured_data.csv", parse_dates=["timestamp"])
    print(f"Dataset: {len(df)} hourly rows, "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Target: measured PM2.5 (median of Karachi ground monitors)")
    print(f"Features: {len(MODEL_FEATURES)}")
    print("Evaluation: rolling-origin, 5 folds, every season tested\n")

    all_results = {}
    selection = None

    for horizon in REPORT_HORIZONS:
        results = evaluate_horizon(df, horizon)
        all_results[f"{horizon}hr"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "per_fold"}
            for k, v in results.items()
        }

        best_key = min(
            (k for k in results if not k.endswith("(baseline)")),
            key=lambda k: results[k]["MAE"],
        )
        print_fold_table(
            results[best_key]["per_fold"],
            results[best_key],
            f"{horizon}-hour horizon — best model: {best_key}",
        )
        print(f"  {'model':<32}{'MAE':>8}{'R2':>8}{'vs persist':>12}{'vs CAMS':>10}")
        for name, m in results.items():
            print(
                f"  {name:<32}{m['MAE']:>8.2f}{m['R2']:>8.3f}"
                f"{m['skill_vs_persistence']:>11.1%}"
                f"{(m.get('skill_vs_cams') or 0):>10.1%}"
            )
        print()

        if horizon == SELECTION_HORIZON:
            selection = results

    algorithms = {
        k: v for k, v in selection.items() if not k.endswith("(baseline)")
    }
    best_name = min(algorithms, key=lambda k: algorithms[k]["MAE"])
    best = algorithms[best_name]

    print("=" * 68)
    print(f"Selected at the {SELECTION_HORIZON}-hour horizon: {best_name}")
    print(f"  MAE {best['MAE']} ug/m3   R2 {best['R2']}")
    print(f"  {best['skill_vs_persistence']:.1%} better than persistence")
    print(f"  {best['skill_vs_cams']:.1%} better than the CAMS forecast")
    print("=" * 68)

    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)
    with open("models/all_results.json", "w") as f:
        json.dump(
            {
                "target": "measured PM2.5 (OpenAQ ground monitors, city median)",
                "evaluation": "rolling-origin backtest, 5 folds",
                "selection_horizon_hours": SELECTION_HORIZON,
                "n_features": len(MODEL_FEATURES),
                "n_rows": int(len(df)),
                "period_start": str(df["timestamp"].min()),
                "period_end": str(df["timestamp"].max()),
                "best_model": best_name,
                "by_horizon": all_results,
            },
            f,
            indent=2,
        )
    print("\nWrote models/all_results.json and models/best_model_name.txt")


if __name__ == "__main__":
    main()

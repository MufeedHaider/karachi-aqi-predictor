"""
Shared evaluation machinery: baselines, metrics and rolling-origin backtesting.

Why rolling-origin rather than one split
----------------------------------------
A single chronological 80/20 split on a year of Karachi data puts the entire
test set in June-September — the low-pollution season, when CAMS happens to be
accurate and the air is calm. A model scored that way is never tested on
December, when measured PM2.5 triples and the forecast actually matters.

Rolling-origin backtesting fixes that. The data is cut at several points; each
time the model trains on everything before the cut and is scored on the window
after it. Every season ends up in a test set exactly once, and training never
sees anything from its own future.

Baselines
---------
Reporting MAE alone says nothing. These give it meaning:

  persistence  — "PM2.5 in h hours equals PM2.5 now". Hard to beat at short
                 leads because the series is smooth.
  climatology  — the training-period average for that hour of day.
  CAMS         — what the Copernicus model itself predicts for that hour. This
                 is the interesting one: it is a real operational forecast, so
                 beating it means the model adds something over the physics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fractions of the record at which to cut. The first is the smallest training
# set; each subsequent window is tested by a model trained on everything before.
DEFAULT_FOLDS = [0.45, 0.56, 0.67, 0.78, 0.89]


def metrics(y_true, y_pred, persistence, cams=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    persistence_mae = mean_absolute_error(y_true, persistence)

    out = {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
        "skill_vs_persistence": round(float(1 - mae / persistence_mae), 3)
        if persistence_mae > 0 else None,
    }
    if cams is not None:
        cams_mae = mean_absolute_error(y_true, cams)
        out["cams_MAE"] = round(float(cams_mae), 3)
        out["skill_vs_cams"] = round(float(1 - mae / cams_mae), 3) if cams_mae > 0 else None
    return out


def climatology_map(target_hours, values):
    """Mean measured PM2.5 per hour of day, from training rows only."""
    import pandas as pd

    return pd.Series(np.asarray(values)).groupby(np.asarray(target_hours)).mean()


def rolling_origin(X, y_delta, current, target_ts, fit_predict,
                   folds=DEFAULT_FOLDS, cams=None, min_test_rows=200):
    """Backtest across seasons.

    `fit_predict(X_train, d_train, X_test)` returns predicted *deltas* for the
    test rows. Returns (per_fold, combined) where combined pools every fold's
    predictions so the headline number covers the whole year.
    """
    n = len(X)
    per_fold = []
    pooled = {"y": [], "pred": [], "persist": [], "cams": []}

    for i, start_frac in enumerate(folds):
        start = int(n * start_frac)
        end = int(n * (folds[i + 1] if i + 1 < len(folds) else 1.0))
        if end - start < min_test_rows:
            continue

        pred_delta = fit_predict(
            X.iloc[:start], y_delta.iloc[:start], X.iloc[start:end]
        )
        cur_test = current.iloc[start:end].to_numpy()
        y_true = cur_test + y_delta.iloc[start:end].to_numpy()
        y_pred = np.clip(np.asarray(pred_delta) + cur_test, 0, None)

        fold_cams = None if cams is None else np.asarray(cams.iloc[start:end])

        fold = metrics(y_true, y_pred, cur_test, fold_cams)
        fold["window"] = (
            f"{target_ts.iloc[start]:%b %d} - {target_ts.iloc[end - 1]:%b %d}"
        )
        fold["n_test"] = int(end - start)
        fold["mean_observed"] = round(float(y_true.mean()), 2)
        per_fold.append(fold)

        pooled["y"] += list(y_true)
        pooled["pred"] += list(y_pred)
        pooled["persist"] += list(cur_test)
        if fold_cams is not None:
            pooled["cams"] += list(fold_cams)

    if not pooled["y"]:
        return [], {}

    combined = metrics(
        pooled["y"],
        pooled["pred"],
        pooled["persist"],
        pooled["cams"] if pooled["cams"] else None,
    )
    combined["n_test"] = len(pooled["y"])
    combined["n_folds"] = len(per_fold)
    combined["mean_observed"] = round(float(np.mean(pooled["y"])), 2)
    return per_fold, combined


def print_fold_table(per_fold, combined, title):
    print(f"--- {title} ---")
    header = f"  {'test window':<22}{'MAE':>7}{'persist':>9}{'CAMS':>8}{'skill':>8}{'vs CAMS':>9}{'observed':>10}"
    print(header)
    for f in per_fold:
        print(
            f"  {f['window']:<22}{f['MAE']:>7.2f}{f['MAE'] / (1 - f['skill_vs_persistence']):>9.2f}"
            f"{f.get('cams_MAE', float('nan')):>8.2f}"
            f"{f['skill_vs_persistence']:>7.1%}{f.get('skill_vs_cams', 0):>9.1%}"
            f"{f['mean_observed']:>10.1f}"
        )
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'ALL FOLDS':<22}{combined['MAE']:>7.2f}"
        f"{combined['MAE'] / (1 - combined['skill_vs_persistence']):>9.2f}"
        f"{combined.get('cams_MAE', float('nan')):>8.2f}"
        f"{combined['skill_vs_persistence']:>7.1%}{combined.get('skill_vs_cams', 0):>9.1%}"
        f"{combined['mean_observed']:>10.1f}"
    )
    print(f"  R2 across all folds: {combined['R2']:.3f}\n")

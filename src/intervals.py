"""
80% prediction range for every forecast (conformalised quantile regression).

A P10/P90 XGBoost model per horizon is fitted on the older part of the training
window, then widened on the newest part so the band really holds ~80% of
outcomes. Romano, Patterson & Candes (2019).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb

ALPHA = 0.2                    # 80% interval
QUANTILES = (ALPHA / 2, 1 - ALPHA / 2)
CAL_FRACTION = 0.2             # tail of the training window used to calibrate


def make_quantile_model():
    """Smaller than the point model: 72 of these also have to fit inside CI."""
    return xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.array(QUANTILES),
        n_estimators=200,
        learning_rate=0.06,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


@dataclass
class IntervalModel:
    model: xgb.XGBRegressor
    correction: float          # conformal widening, in ug/m3 (can be negative)
    n_calibration: int

    def predict(self, X, current):
        """Return (lo, hi) in PM2.5 units, clipped at zero."""
        q = np.asarray(self.model.predict(X))
        q = q.reshape(len(X), -1)
        lo = np.asarray(current, float) + q[:, 0] - self.correction
        hi = np.asarray(current, float) + q[:, -1] + self.correction
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)   # quantile crossing
        return np.clip(lo, 0, None), np.clip(hi, 0, None)


def conformal_correction(lo, hi, y, alpha=ALPHA):
    """Finite-sample CQR correction from a calibration set."""
    y = np.asarray(y, float)
    scores = np.maximum(np.asarray(lo) - y, y - np.asarray(hi))
    n = len(scores)
    level = min(1.0, (1 - alpha) * (n + 1) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_interval_model(X, y_delta, current, cal_fraction=CAL_FRACTION):
    """Fit quantile model on the head of the window, calibrate on the tail.

    Chronological split: the calibration rows are strictly later than the
    rows the quantile model learned from, as the forecast rows will be.
    """
    n = len(X)
    cut = int(n * (1 - cal_fraction))
    model = make_quantile_model()
    model.fit(X.iloc[:cut], y_delta.iloc[:cut])

    provisional = IntervalModel(model, 0.0, 0)
    lo, hi = provisional.predict(X.iloc[cut:], current.iloc[cut:])
    y_cal = current.iloc[cut:].to_numpy() + y_delta.iloc[cut:].to_numpy()
    corr = conformal_correction(lo, hi, y_cal)
    return IntervalModel(model, corr, n - cut)


def pinball(y, q_pred, q):
    diff = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_metrics(y, lo, hi):
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    return {
        "coverage": round(float(np.mean((y >= lo) & (y <= hi))), 3),
        "mean_width": round(float(np.mean(hi - lo)), 3),
        "pinball_p10": round(pinball(y, lo, QUANTILES[0]), 3),
        "pinball_p90": round(pinball(y, hi, QUANTILES[1]), 3),
    }


def backtest_intervals(X, y_delta, current, target_ts, folds, min_test_rows=200):
    """Rolling-origin backtest of the interval, on the same folds as the point model."""
    n = len(X)
    per_fold, pooled = [], {"y": [], "lo": [], "hi": []}
    for i, start_frac in enumerate(folds):
        start = int(n * start_frac)
        end = int(n * (folds[i + 1] if i + 1 < len(folds) else 1.0))
        if end - start < min_test_rows:
            continue
        im = fit_interval_model(X.iloc[:start], y_delta.iloc[:start], current.iloc[:start])
        lo, hi = im.predict(X.iloc[start:end], current.iloc[start:end])
        y = current.iloc[start:end].to_numpy() + y_delta.iloc[start:end].to_numpy()
        fold = interval_metrics(y, lo, hi)
        fold["window"] = f"{target_ts.iloc[start]:%b %d} - {target_ts.iloc[end - 1]:%b %d}"
        fold["conformal_correction"] = round(im.correction, 3)
        per_fold.append(fold)
        pooled["y"] += list(y)
        pooled["lo"] += list(lo)
        pooled["hi"] += list(hi)
    if not pooled["y"]:
        return [], {}
    combined = interval_metrics(pooled["y"], pooled["lo"], pooled["hi"])
    combined["nominal_coverage"] = 1 - ALPHA
    combined["n_test"] = len(pooled["y"])
    return per_fold, combined

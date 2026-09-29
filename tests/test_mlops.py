"""Tests for the v2 additions: seasonal baseline, intervals, gate, drift, live scoring.

All synthetic, no network, no API key: they run in CI before the retrain.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import accuracy_gate  # noqa: E402
from drift import compare, drift_report  # noqa: E402
from evaluation import seasonal_naive  # noqa: E402
from intervals import (  # noqa: E402
    conformal_correction,
    fit_interval_model,
    interval_metrics,
)
from live_scoring import dedupe, forecast_to_archive_rows, score  # noqa: E402


# ------------------------------------------------------------ seasonal naive

def _series_with_gap():
    ts = pd.date_range("2026-01-01", periods=24 * 5, freq="h")
    df = pd.DataFrame({"timestamp": ts, "ground_pm25": np.arange(len(ts), dtype=float)})
    return df.drop(index=[30, 31]).reset_index(drop=True)   # rows missing, like real data


@pytest.mark.parametrize("h, expected_lag", [(1, 23), (24, 0), (25, 23), (48, 0), (30, 18)])
def test_seasonal_naive_uses_same_hour_of_day_at_or_before_t(h, expected_lag):
    df = _series_with_gap()
    idx = df.index[df["timestamp"] >= "2026-01-04"]
    pred = seasonal_naive(df, idx, h)
    lookup = df.loc[idx, "timestamp"] - pd.Timedelta(hours=expected_lag)
    # value is the hour number, so it can be checked directly
    expected = (lookup - df["timestamp"].iloc[0]).dt.total_seconds() / 3600
    assert np.allclose(pred.to_numpy(), expected.to_numpy())
    # target hour and predicted-from hour share an hour of day
    target = df.loc[idx, "timestamp"] + pd.Timedelta(hours=h)
    assert (target.dt.hour == lookup.dt.hour).all()


def test_seasonal_naive_is_by_timestamp_not_row_offset():
    df = _series_with_gap()
    row = df.index[df["timestamp"] == pd.Timestamp("2026-01-02 08:00")][0]
    # 24 rows back crosses the two dropped rows, so it would be the wrong hour
    pred = seasonal_naive(df, pd.Index([row]), 24)
    assert pred.iloc[0] == df.loc[row, "ground_pm25"]      # lag 0 at h=24


def test_seasonal_naive_never_looks_past_t():
    df = _series_with_gap()
    idx = df.index[df["timestamp"] >= "2026-01-03"]
    for h in range(1, 73):
        pred = seasonal_naive(df, idx, h)
        ok = pred.notna()
        assert (pred[ok] <= df.loc[idx[ok], "ground_pm25"]).all()


# ------------------------------------------------------------ intervals

def test_conformal_correction_reaches_nominal_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)       # too narrow
    corr = conformal_correction(lo[:2500], hi[:2500], y[:2500], alpha=0.2)
    covered = np.mean((y[2500:] >= lo[2500:] - corr) & (y[2500:] <= hi[2500:] + corr))
    assert 0.77 <= covered <= 0.83
    assert corr > 0


def test_interval_model_is_calibrated_on_unseen_data():
    rng = np.random.default_rng(2)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(0, 2, size=n)})
    noise = rng.normal(0, 1, n) * (0.5 + X["b"])                 # heteroscedastic
    delta = pd.Series(2 * X["a"] + noise)
    current = pd.Series(np.full(n, 50.0))
    im = fit_interval_model(X.iloc[:2000], delta.iloc[:2000], current.iloc[:2000])
    lo, hi = im.predict(X.iloc[2000:], current.iloc[2000:])
    y = current.iloc[2000:] + delta.iloc[2000:]
    m = interval_metrics(y, lo, hi)
    assert 0.72 <= m["coverage"] <= 0.88
    assert (lo <= hi).all() and (lo >= 0).all()
    # heteroscedastic noise: the band should be wider where b is larger
    wide = X["b"].iloc[2000:].to_numpy() > 1.5
    narrow = X["b"].iloc[2000:].to_numpy() < 0.5
    assert (hi - lo)[wide].mean() > (hi - lo)[narrow].mean()


# ------------------------------------------------------------ accuracy gate

def _results(cams=0.40, pers=0.18, mae24=5.8, coverage=0.80):
    return {
        "mean_skill_vs_cams": cams,
        "mean_skill_vs_persistence": pers,
        "horizons": {"24": {"MAE": mae24}},
        "intervals": {"24": {"coverage": coverage}},
    }


def test_gate_passes_a_stable_model():
    checks = accuracy_gate.evaluate(_results(), _results(cams=0.41, mae24=5.7))
    assert all(ok for _, ok, _ in checks)


@pytest.mark.parametrize("new", [
    _results(cams=-0.01),              # lost to CAMS
    _results(pers=-0.02),              # lost to persistence
    _results(cams=0.30),               # 10-point drop vs previous
    _results(mae24=7.0),               # 24h MAE up 20%
])
def test_gate_blocks_regressions(new):
    checks = accuracy_gate.evaluate(new, _results())
    assert not all(ok for _, ok, _ in checks)


def test_gate_without_history_still_checks_baselines():
    assert all(ok for _, ok, _ in accuracy_gate.evaluate(_results(), None))
    assert not all(ok for _, ok, _ in accuracy_gate.evaluate(_results(cams=-0.1), None))


def test_bad_interval_coverage_warns_but_never_blocks():
    new = _results(coverage=0.55)
    assert all(ok for _, ok, _ in accuracy_gate.evaluate(new, _results()))
    assert not all(ok for _, ok, _ in accuracy_gate.warnings(new))


# ------------------------------------------------------------ drift

def test_same_distribution_is_stable():
    rng = np.random.default_rng(3)
    assert compare(rng.normal(30, 8, 5000), rng.normal(30, 8, 168))["level"] == "stable"


def test_values_beyond_training_range_are_major():
    rng = np.random.default_rng(4)
    out = compare(rng.normal(30, 8, 5000), rng.normal(90, 8, 168))
    assert out["level"] == "major" and out["out_of_range"] > 0.9


def test_a_calmer_week_is_not_drift():
    """Narrower spread inside the usual range is not a new regime."""
    rng = np.random.default_rng(5)
    assert compare(rng.normal(30, 10, 5000), rng.normal(27, 4, 168))["level"] == "stable"


def test_drift_compares_with_the_same_season_last_year():
    ts = pd.date_range("2025-01-01", "2026-03-01", freq="h")
    rng = np.random.default_rng(6)
    season = 50 + 30 * np.cos(2 * np.pi * ts.dayofyear.to_numpy() / 365.25)
    df = pd.DataFrame({"timestamp": ts, "ground_pm25": season + rng.normal(0, 3, len(ts))})
    rep = drift_report(df, columns=["ground_pm25"])
    assert rep["reference"] == "same weeks last year"
    assert rep["features"]["ground_pm25"]["level"] == "stable"


def test_drift_falls_back_without_a_year_of_history():
    ts = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    df = pd.DataFrame({"timestamp": ts, "ground_pm25": np.r_[np.full(24 * 53, 20.0), np.full(24 * 7, 80.0)]
                       + np.random.default_rng(7).normal(0, 1, len(ts))})
    rep = drift_report(df, columns=["ground_pm25"])
    assert rep["reference"] == "all earlier data"
    assert rep["features"]["ground_pm25"]["level"] == "major"


# ------------------------------------------------------------ live scoring

def _published(anchor, preds, cams):
    ts = pd.date_range(pd.Timestamp(anchor) + pd.Timedelta(hours=1), periods=len(preds), freq="h")
    return pd.DataFrame({
        "timestamp": ts, "horizon_hour": range(1, len(preds) + 1),
        "pm2_5_predicted": preds, "aqi_predicted": 0, "aqi_category": "x",
        "cams_pm2_5": cams,
    })


def test_anchor_is_recovered_from_the_forecast_file():
    rows = forecast_to_archive_rows(_published("2026-09-01 23:00", [1, 2, 3], [1, 1, 1]),
                                    "2026-09-02T04:00:00+00:00")
    assert (rows["anchor"] == pd.Timestamp("2026-09-01 23:00")).all()
    assert rows["pm2_5_p10"].isna().all()     # older files had no interval


def test_reissued_forecast_does_not_overwrite_the_first():
    first = forecast_to_archive_rows(_published("2026-09-01 23:00", [10] * 3, [5] * 3),
                                     "2026-09-02T04:00:00+00:00")
    later = forecast_to_archive_rows(_published("2026-09-01 23:00", [99] * 3, [5] * 3),
                                     "2026-09-02T09:00:00+00:00")
    kept = dedupe(pd.concat([later, first]))
    assert len(kept) == 3 and (kept["pm2_5_predicted"] == 10).all()


def test_live_score_uses_anchor_measurement_as_persistence():
    anchor = pd.Timestamp("2026-09-01 23:00")
    archive = forecast_to_archive_rows(
        _published(anchor, [21.0] * 24, [10.0] * 24), "2026-09-02T04:00:00+00:00")
    ts = pd.date_range(anchor, periods=25, freq="h")
    ground = pd.DataFrame({"timestamp": ts, "pm2_5": [30.0] + [20.0] * 24})
    out = score(archive, ground)
    o = out["overall"]
    assert o["MAE"] == pytest.approx(1.0)
    assert o["cams_MAE"] == pytest.approx(10.0)
    assert o["persistence_MAE"] == pytest.approx(10.0)
    assert o["skill_vs_cams"] == pytest.approx(0.9)
    assert out["median_publish_latency_hours"] == pytest.approx(5.0)

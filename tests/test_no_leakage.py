"""Tests that the model forecasts rather than reads the answer.

The bug these exist to prevent: the project originally trained on `y = pm2_5[t]`
while feeding the model `pm_ratio = pm2_5[t] / pm10[t]` and
`wind_dispersion = wind_speed[t] / pm2_5[t]`. Both are invertible, so the target
was recoverable from the inputs to machine precision, and the reported R² of
0.978 measured nothing but that.

The structural fix is that the label lives at t + horizon and is a *measured*
value from ground monitors, while every feature is observable at t. These tests
check that this stays true.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import (  # noqa: E402
    BANNED_AS_FEATURES,
    BASE_FEATURES,
    FUTURE_FEATURES,
    MODEL_FEATURES,
    build_supervised,
    engineer_features,
)


@pytest.fixture(scope="module")
def frame():
    """Synthetic hourly data — no network, no cached CSV, no API key."""
    rng = np.random.default_rng(0)
    n = 24 * 150
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    hour = ts.hour.to_numpy()

    # Measured PM2.5: diurnal cycle plus autocorrelated noise.
    trend = 45 + 12 * np.cos(2 * np.pi * hour / 24)
    noise = pd.Series(rng.normal(0, 4, n)).rolling(6, min_periods=1).mean().to_numpy()
    ground = np.clip(trend + noise, 1, None)

    # CAMS reads low, as it does in reality.
    cams = ground * 0.62 + rng.normal(0, 2, n)

    raw = pd.DataFrame(
        {
            "timestamp": ts,
            "pm2_5": np.clip(cams, 1, None),
            "pm10": np.clip(cams * 1.7, 1, None),
            "no2": rng.uniform(5, 60, n),
            "ozone": rng.uniform(10, 90, n),
            "co": rng.uniform(100, 600, n),
            "so2": rng.uniform(1, 30, n),
            "temperature": 25 + 6 * np.sin(2 * np.pi * hour / 24),
            "humidity": rng.uniform(30, 90, n),
            "wind_speed": rng.uniform(1, 20, n),
            "wind_dir": rng.uniform(0, 360, n),
            "pressure": rng.uniform(1000, 1020, n),
            "ground_pm25": ground,
            "n_stations": rng.integers(4, 9, n),
        }
    )
    return engineer_features(raw, out_path=None)


def test_target_columns_are_never_features():
    assert not BANNED_AS_FEATURES.intersection(MODEL_FEATURES)


def test_no_raw_ground_column_in_features():
    """The measured value at time t is the thing being forecast, not an input."""
    assert "ground_pm25" not in MODEL_FEATURES


@pytest.mark.parametrize("horizon", [1, 6, 24, 72])
def test_label_is_strictly_in_the_future(frame, horizon):
    X, y_delta, current, target_ts = build_supervised(frame, horizon)
    reconstructed = current + y_delta
    expected = frame["ground_pm25"].shift(-horizon).loc[X.index]
    assert np.allclose(reconstructed, expected)


@pytest.mark.parametrize("horizon", [1, 24, 72])
def test_target_timestamp_is_exactly_horizon_hours_later(frame, horizon):
    """After reindexing to an hourly grid, a shift must be a true clock offset."""
    X, _, _, target_ts = build_supervised(frame, horizon)
    gap = target_ts - frame["timestamp"].loc[X.index]
    assert (gap == pd.Timedelta(hours=horizon)).all()


@pytest.mark.parametrize("horizon", [1, 24, 72])
def test_no_feature_reconstructs_the_label(frame, horizon):
    """No single feature may correlate almost perfectly with the label.

    That signature is how pm_ratio and wind_dispersion behaved against the old
    same-hour target.
    """
    X, y_delta, current, _ = build_supervised(frame, horizon)
    y_level = current + y_delta
    for col in X.columns:
        if X[col].std() == 0:
            continue
        r = np.corrcoef(X[col], y_level)[0, 1]
        assert abs(r) < 0.999, f"{col} correlates {r:.6f} with the label"


def test_the_original_leak_would_still_be_caught(frame):
    """Sanity check on the detection itself.

    Against a *same-hour* target, wind_dispersion inverts to reproduce it
    exactly. If this ever stops holding, the test above is not sensitive to the
    bug it exists to catch.
    """
    reconstructed = frame["wind_speed"] / frame["wind_dispersion"] - 1e-6
    assert np.allclose(reconstructed, frame["ground_pm25"], atol=1e-6)


def test_cams_is_a_feature_not_the_target(frame):
    """CAMS must inform the model without becoming what it predicts."""
    assert "cams_pm25" in BASE_FEATURES
    assert "fut_cams_pm25" in FUTURE_FEATURES
    X, y_delta, current, _ = build_supervised(frame, 24)
    y_level = current + y_delta
    # CAMS should be correlated but nowhere near identical.
    r = np.corrcoef(X["fut_cams_pm25"], y_level)[0, 1]
    assert 0.2 < abs(r) < 0.999


def test_outlier_caps_come_from_training_data_only(frame):
    n_train = int(len(frame) * 0.8)
    assert frame["cams_pm25"].max() <= frame["cams_pm25"].iloc[:n_train].max() + 1e-9


def test_feature_counts_are_stable():
    """Guards the README against drifting away from the code."""
    assert len(BASE_FEATURES) == 39
    assert len(FUTURE_FEATURES) == 9
    assert len(MODEL_FEATURES) == 48
    assert len(set(MODEL_FEATURES)) == len(MODEL_FEATURES)

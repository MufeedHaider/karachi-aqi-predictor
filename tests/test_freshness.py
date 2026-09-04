"""Tests for the live-reading freshness guard.

The bug these exist to prevent: the WAQI feed for the Karachi US Consulate was
observed returning a measurement timestamped 2025-03-04 during a run on
2026-09-04 — eighteen months stale. The station had stopped reporting but the
API kept serving its last value, and the dashboard displayed it as the current
air quality.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from freshness import describe_age, is_stale, reading_age_hours  # noqa: E402

ARCHIVE = pd.Timestamp("2026-09-03 23:00:00")


def test_the_exact_reading_that_was_wrong():
    """2025-03-04 against a 2026-09-03 archive must be rejected."""
    assert is_stale(pd.Timestamp("2025-03-04 16:00:00"), ARCHIVE)
    assert "months behind" in describe_age(pd.Timestamp("2025-03-04 16:00:00"), ARCHIVE)


def test_healthy_station_is_ahead_of_the_archive():
    """The archive lags ~a day, so a live reading is normally newer."""
    fresh = ARCHIVE + pd.Timedelta(hours=20)
    assert not is_stale(fresh, ARCHIVE)
    assert reading_age_hours(fresh, ARCHIVE) < 0


def test_boundary():
    assert not is_stale(ARCHIVE - pd.Timedelta(hours=5, minutes=59), ARCHIVE)
    assert is_stale(ARCHIVE - pd.Timedelta(hours=6, minutes=1), ARCHIVE)


def test_missing_timestamp_is_treated_as_stale():
    """If we cannot tell how old a reading is, we must not present it as now."""
    for bad in [None, pd.NaT, "not a date"]:
        assert is_stale(bad, ARCHIVE)


@pytest.mark.parametrize(
    "lag,expected",
    [
        (pd.Timedelta(minutes=20), "current"),
        (pd.Timedelta(hours=9), "9 hours behind"),
        (pd.Timedelta(days=5), "5 days behind"),
    ],
)
def test_describe_age_phrasing(lag, expected):
    assert describe_age(ARCHIVE - lag, ARCHIVE) == expected

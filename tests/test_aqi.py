"""Tests for the EPA AQI conversion.

The bug these exist to prevent: the breakpoint tables were previously written as
explicit (low, high) pairs copied from the EPA publication — 0-9.0, 9.1-35.4,
35.5-55.4 and so on. Those printed bounds are rounded to reporting precision, so
a lookup of the form `if lo <= c <= hi` left a gap between every pair of bands.
A PM10 reading of 54.9 matched nothing and fell through to a `return 500`
fallback, labelling ordinary air as Hazardous. 125 rows of the training set —
1.4% — carried a fabricated AQI of 500 as a result.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aqi import (  # noqa: E402
    aqi_from_no2,
    aqi_from_pm10,
    aqi_from_pm25,
    category,
    overall_aqi,
)


@pytest.mark.parametrize("fn", [aqi_from_pm25, aqi_from_pm10])
def test_no_gaps_between_bands(fn):
    """Sweep the whole reportable range; nothing may fall through to 500."""
    c = 0.0
    while c <= 300.0:
        value = fn(c)
        assert value is not None, f"{fn.__name__}({c}) returned None"
        if c < 200:
            assert value < 500, (
                f"{fn.__name__}({c}) returned {value}: concentration fell "
                "between two bands and hit the fallback"
            )
        c += 0.05


def test_the_exact_value_that_used_to_break():
    """PM10 54.9 sat in the old 54-to-55 gap and was reported as Hazardous."""
    assert aqi_from_pm10(54.9) == pytest.approx(50, abs=1)
    assert category(aqi_from_pm10(54.9)) in {"Good", "Moderate"}


@pytest.mark.parametrize(
    "pm25,expected",
    [
        (0.0, 0),
        (9.0, 50),      # top of Good, 2024 revision
        (35.4, 100),    # top of Moderate
        (55.4, 150),
        (125.4, 200),
        (225.4, 300),
    ],
)
def test_pm25_band_edges_2024(pm25, expected):
    assert aqi_from_pm25(pm25) == pytest.approx(expected, abs=1)


def test_pm25_uses_the_2024_scale_not_2012():
    """Under the old table 12.0 was the Good/Moderate boundary; under the 2024
    revision it is 9.0. A value of 10 must therefore be Moderate, not Good."""
    assert aqi_from_pm25(10.0) > 50
    assert category(aqi_from_pm25(10.0)) == "Moderate"


def test_monotonic():
    """More pollution can never produce a lower AQI."""
    previous = -1
    c = 0.0
    while c <= 250:
        value = aqi_from_pm25(c)
        assert value >= previous, f"AQI decreased at {c}"
        previous = value
        c += 0.1


def test_overall_is_max_of_subindices():
    assert overall_aqi(pm25=100, pm10=10, no2_ugm3=10) == aqi_from_pm25(100)
    assert overall_aqi(pm25=5, pm10=300, no2_ugm3=10) == aqi_from_pm10(300)


def test_missing_pollutants_are_skipped_not_zeroed():
    """A missing pollutant must not be silently treated as a reading of zero."""
    assert overall_aqi(pm25=50, pm10=None, no2_ugm3=None) == aqi_from_pm25(50)
    assert overall_aqi(pm25=None, pm10=None, no2_ugm3=None) is None


def test_handles_junk_input():
    for junk in [None, float("nan"), "abc"]:
        assert aqi_from_pm25(junk) is None


def test_negative_clamped_not_crashed():
    assert aqi_from_pm25(-5) == 0


def test_no2_converts_ugm3_to_ppb():
    """NO2 bands are in ppb; input is µg/m³."""
    assert aqi_from_no2(53 * 1.88) == pytest.approx(50, abs=1)

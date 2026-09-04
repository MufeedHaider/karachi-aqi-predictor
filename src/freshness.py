"""
Freshness checks for live readings.

A ground station can go quiet without saying so: the WAQI feed keeps serving
its last successful measurement, with the original timestamp attached. The
Karachi US Consulate feed was observed returning a reading over a year stale.

Displaying that as "current air quality" is worse than showing nothing, so the
dashboard checks age before trusting a reading.

Comparing against the wall clock would be fragile here — the station reports in
Karachi local time while a cloud host runs in UTC, which would make every
reading look five hours older than it is. Instead the reading is compared to the
newest observation already in the dataset, which is in the same timezone. A
healthy station is *ahead* of the archive (the archive lags roughly a day), so
a reading that sits behind it is a clear signal the feed has stopped.
"""

from __future__ import annotations

import pandas as pd

# How far behind the newest archived observation a station reading may fall
# before it is treated as stale.
DEFAULT_MAX_LAG_HOURS = 6


def reading_age_hours(reading_ts, reference_ts):
    """Hours by which `reading_ts` lags `reference_ts`.

    Negative means the reading is newer than the reference, which is the normal,
    healthy case. Returns None if either timestamp is unusable.
    """
    try:
        reading = pd.to_datetime(reading_ts)
        reference = pd.to_datetime(reference_ts)
    except (ValueError, TypeError):
        return None
    if pd.isna(reading) or pd.isna(reference):
        return None
    return (reference - reading).total_seconds() / 3600.0


def is_stale(reading_ts, reference_ts, max_lag_hours=DEFAULT_MAX_LAG_HOURS):
    """True when a live reading is too far behind to be shown as current.

    A reading with no usable timestamp is treated as stale: if we cannot tell
    how old it is, we do not present it as now.
    """
    age = reading_age_hours(reading_ts, reference_ts)
    if age is None:
        return True
    return age > max_lag_hours


def describe_age(reading_ts, reference_ts):
    """Short human phrase for how far behind a reading is."""
    age = reading_age_hours(reading_ts, reference_ts)
    if age is None:
        return "age unknown"
    if age < 1:
        return "current"
    if age < 48:
        return f"{age:.0f} hours behind"
    days = age / 24
    if days < 60:
        return f"{days:.0f} days behind"
    return f"{days / 30.4:.0f} months behind"

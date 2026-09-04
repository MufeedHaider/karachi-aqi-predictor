"""
US EPA Air Quality Index — single source of truth for the whole project.

Every module imports from here. Previously each script carried its own copy of
the breakpoint tables, which drifted apart: the forecaster was still using the
pre-2024 PM2.5 table while the dashboard used the current one, so the same
concentration produced two different AQI values on two pages of the same app.

Breakpoints are the US EPA tables effective 6 May 2024 (PM2.5 revised; PM10 and
NO2 unchanged). Reference: 40 CFR Part 58, Appendix G.

Implementation note — the tables below store only the UPPER bound of each band.
An earlier version stored explicit (low, high) pairs copied from the EPA
publication, which prints them as 0-9.0, 9.1-35.4, 35.5-55.4 ... Those printed
bounds are truncated to reporting precision, so a lookup written as
`if lo <= c <= hi` left real gaps: a PM10 reading of 54.9 matched no band and
fell through to the "return 500" fallback, silently labelling ordinary air as
Hazardous. Using upper bounds with a running lower bound makes the bands
exhaustive by construction, so no value can fall through.
"""

from __future__ import annotations

# (upper concentration bound, AQI low, AQI high)
# The lower concentration bound of each band is the previous band's upper bound.
PM25_BANDS = [
    (9.0, 0, 50),
    (35.4, 51, 100),
    (55.4, 101, 150),
    (125.4, 151, 200),
    (225.4, 201, 300),
    (325.4, 301, 500),
]

PM10_BANDS = [
    (54, 0, 50),
    (154, 51, 100),
    (254, 101, 150),
    (354, 151, 200),
    (424, 201, 300),
    (604, 301, 500),
]

# NO2 bands are defined in parts per billion, not µg/m³.
NO2_BANDS_PPB = [
    (53, 0, 50),
    (100, 51, 100),
    (360, 101, 150),
    (649, 151, 200),
    (1249, 201, 300),
    (2049, 301, 500),
]

# EPA conversion at 25 °C / 1 atm: 1 ppb NO2 = 1.88 µg/m³
NO2_UGM3_PER_PPB = 1.88

CATEGORIES = [
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (10**9, "Hazardous"),
]


def _piecewise(concentration, bands):
    """Linear interpolation within the EPA band containing `concentration`.

    Bands are treated as contiguous: band i covers (upper[i-1], upper[i]], with
    the first band starting at 0. Values above the top band are clamped to 500,
    which is the EPA maximum reportable AQI.
    """
    if concentration is None:
        return None
    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return None
    if c != c:  # NaN
        return None
    if c < 0:
        c = 0.0

    lo_c = 0.0
    for hi_c, lo_aqi, hi_aqi in bands:
        if c <= hi_c:
            span = hi_c - lo_c
            if span <= 0:
                return float(lo_aqi)
            return round(((hi_aqi - lo_aqi) / span) * (c - lo_c) + lo_aqi)
        lo_c = hi_c
    return 500


def aqi_from_pm25(pm25):
    """AQI sub-index for a PM2.5 concentration in µg/m³."""
    return _piecewise(pm25, PM25_BANDS)


def aqi_from_pm10(pm10):
    """AQI sub-index for a PM10 concentration in µg/m³."""
    return _piecewise(pm10, PM10_BANDS)


def aqi_from_no2(no2_ugm3):
    """AQI sub-index for an NO2 concentration given in µg/m³."""
    if no2_ugm3 is None:
        return None
    try:
        ppb = float(no2_ugm3) / NO2_UGM3_PER_PPB
    except (TypeError, ValueError):
        return None
    return _piecewise(ppb, NO2_BANDS_PPB)


def overall_aqi(pm25=None, pm10=None, no2_ugm3=None):
    """Overall AQI = the highest sub-index across reported pollutants (EPA rule).

    Pollutants that are missing are skipped rather than treated as zero.
    Returns None if no pollutant could be evaluated.
    """
    subs = [
        aqi_from_pm25(pm25),
        aqi_from_pm10(pm10),
        aqi_from_no2(no2_ugm3),
    ]
    subs = [s for s in subs if s is not None]
    return max(subs) if subs else None


def category(aqi):
    """EPA category name for an AQI value."""
    if aqi is None:
        return "Unknown"
    for hi, name in CATEGORIES:
        if aqi <= hi:
            return name
    return "Hazardous"


def short_category(aqi):
    """Compact category label, for narrow dashboard cards."""
    name = category(aqi)
    return "Unhealthy (Sensitive)" if name == "Unhealthy for Sensitive Groups" else name


def color(aqi):
    """Official EPA display colour for an AQI value."""
    if aqi is None:
        return "#64748b"
    if aqi <= 50:
        return "#00e676"
    if aqi <= 100:
        return "#ffea00"
    if aqi <= 150:
        return "#ff9100"
    if aqi <= 200:
        return "#ff1744"
    if aqi <= 300:
        return "#d500f9"
    return "#b71c1c"


def advice(aqi):
    """Health guidance matching the EPA category."""
    if aqi is None:
        return "No reading available."
    if aqi <= 50:
        return "Air quality is good. Safe for all outdoor activity."
    if aqi <= 100:
        return "Acceptable. Unusually sensitive people should limit prolonged exertion outdoors."
    if aqi <= 150:
        return "Sensitive groups (asthma, heart or lung conditions, children, elderly) should limit prolonged outdoor exertion."
    if aqi <= 200:
        return "Everyone should reduce prolonged outdoor exertion. Sensitive groups should stay indoors."
    if aqi <= 300:
        return "Health alert. Everyone should avoid outdoor exertion."
    return "Health emergency. Everyone should remain indoors with windows closed."

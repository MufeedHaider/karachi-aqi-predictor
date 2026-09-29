"""
Input drift: is the model being asked about conditions it has not seen?

The last RECENT_DAYS are compared with the same weeks one year earlier, so a
normal seasonal swing is not flagged. With less than a year of history it falls
back to all earlier data and says so. Two plain measures per input:

  out_of_range  share of recent hours outside the reference 1st-99th percentile
                (the model has little training data out there)
  shift         change in the mean, in reference standard deviations

  major     out_of_range > 20%  or  shift > 1.0
  moderate  out_of_range > 5%   or  shift > 0.5
  stable    otherwise

Informational only: it never blocks a publish.
"""

from __future__ import annotations

import json
import os

import pandas as pd

FEATURED = "data/featured_data.csv"
REPORT = "models/drift_report.json"
RECENT_DAYS = 7
SEASON_DAYS = 14
MIN_REFERENCE_ROWS = 24 * 14

WATCH = ["ground_pm25", "cams_pm25", "cams_gap", "wind_speed",
         "humidity", "temperature", "cams_pm10"]


def compare(reference, recent):
    ref = pd.Series(reference, dtype=float).dropna()
    rec = pd.Series(recent, dtype=float).dropna()
    if len(ref) < 50 or rec.empty:
        return {"level": "n/a"}
    lo, hi = ref.quantile([0.01, 0.99])
    outside = float(((rec < lo) | (rec > hi)).mean())
    sd = ref.std()
    shift = float(abs(rec.mean() - ref.mean()) / sd) if sd > 0 else 0.0
    if outside > 0.20 or shift > 1.0:
        level = "major"
    elif outside > 0.05 or shift > 0.5:
        level = "moderate"
    else:
        level = "stable"
    return {
        "level": level,
        "out_of_range": round(outside, 3),
        "shift_sd": round(shift, 2),
        "reference_mean": round(float(ref.mean()), 2),
        "recent_mean": round(float(rec.mean()), 2),
    }


def same_season_last_year(df, start, end, season_days=SEASON_DAYS):
    lo = start - pd.DateOffset(years=1) - pd.Timedelta(days=season_days)
    hi = end - pd.DateOffset(years=1) + pd.Timedelta(days=season_days)
    return df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]


def drift_report(df, recent_days=RECENT_DAYS, columns=WATCH):
    df = df.sort_values("timestamp")
    end = df["timestamp"].max()
    cutoff = end - pd.Timedelta(days=recent_days)
    recent = df[df["timestamp"] > cutoff]
    ref = same_season_last_year(df, cutoff, end)
    reference = "same weeks last year"
    if len(ref) < MIN_REFERENCE_ROWS:
        ref, reference = df[df["timestamp"] <= cutoff], "all earlier data"

    features = {c: compare(ref[c], recent[c]) for c in columns if c in df.columns}
    order = {"n/a": -1, "stable": 0, "moderate": 1, "major": 2}
    overall = max((v["level"] for v in features.values()), key=order.get, default="n/a")
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "recent_window": f"last {recent_days} days",
        "reference": reference,
        "n_recent": int(len(recent)),
        "overall": overall,
        "features": features,
    }


def main():
    if not os.path.exists(FEATURED):
        raise SystemExit(f"{FEATURED} missing: run feature_engineering.py first")
    report = drift_report(pd.read_csv(FEATURED, parse_dates=["timestamp"]))
    os.makedirs("models", exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Input drift ({report['recent_window']} vs {report['reference']}): {report['overall']}")
    for name, v in report["features"].items():
        if v["level"] != "n/a":
            print(f"  {name:<16} {v['level']:<9} outside range {v['out_of_range']:.0%}  "
                  f"shift {v['shift_sd']} sd")


if __name__ == "__main__":
    main()

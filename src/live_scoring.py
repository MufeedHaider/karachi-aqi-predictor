"""
Live accuracy: score every published forecast against what was later measured.

Each nightly run commits data/forecast_72hr.csv. Those files are archived in
data/forecast_archive.csv (--backfill rebuilds it from git history) and scored
against OpenAQ measurements, next to CAMS and persistence.

    python src/live_scoring.py --backfill   # once
    python src/live_scoring.py              # nightly
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ARCHIVE_PATH = "data/forecast_archive.csv"
GROUND_PATH = "data/ground_history.csv"
FORECAST_PATH = "data/forecast_72hr.csv"
RESULTS_PATH = "models/live_results.json"

ARCHIVE_COLS = [
    "issued_at", "anchor", "timestamp", "horizon_hour",
    "pm2_5_predicted", "pm2_5_p10", "pm2_5_p90", "cams_pm2_5",
]

# Horizon buckets for reporting: enough rows per bucket to mean something.
BUCKETS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 72)]


def forecast_to_archive_rows(forecast: pd.DataFrame, issued_at) -> pd.DataFrame:
    """Normalise one published forecast file into archive rows.

    The anchor (last measured hour the forecast was built from) is recovered as
    the first target hour minus its horizon, which holds by construction.
    """
    f = forecast.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    first = f.sort_values("horizon_hour").iloc[0]
    anchor = first["timestamp"] - pd.Timedelta(hours=int(first["horizon_hour"]))
    f["anchor"] = anchor
    f["issued_at"] = pd.to_datetime(issued_at, utc=True).tz_convert(None)
    for col in ["pm2_5_p10", "pm2_5_p90"]:
        if col not in f.columns:
            f[col] = np.nan
    return f[ARCHIVE_COLS]


def _git(*args) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def backfill_from_git(path=FORECAST_PATH) -> pd.DataFrame:
    """Every committed version of the forecast file, as archive rows.

    Only versions with a `cams_pm2_5` column are used: that column arrived with
    the rebuild that made the target measured PM2.5, so older files forecast a
    different quantity and would contaminate the score.
    """
    log = _git("log", "--format=%H %cI", "--", path).strip().splitlines()
    frames = []
    for line in log:
        sha, committed = line.split(" ", 1)
        try:
            raw = _git("show", f"{sha}:{path}")
        except subprocess.CalledProcessError:
            continue
        frame = pd.read_csv(io.StringIO(raw))
        if "cams_pm2_5" not in frame.columns or frame.empty:
            continue
        frames.append(forecast_to_archive_rows(frame, committed))
    if not frames:
        return pd.DataFrame(columns=ARCHIVE_COLS)
    return dedupe(pd.concat(frames, ignore_index=True))


def dedupe(archive: pd.DataFrame) -> pd.DataFrame:
    """Keep one forecast per (anchor, horizon): the first one published.

    Two commits can share an anchor when a run is repeated on the same day. The
    earliest is the one users saw first, so later re-issues do not get to
    overwrite it with hindsight.
    """
    archive = archive.sort_values("issued_at")
    return (
        archive.drop_duplicates(subset=["anchor", "horizon_hour"], keep="first")
        .sort_values(["anchor", "horizon_hour"])
        .reset_index(drop=True)
    )


def append_forecast(forecast: pd.DataFrame, issued_at=None, path=ARCHIVE_PATH):
    """Called by the nightly run after writing a new forecast."""
    issued_at = issued_at or pd.Timestamp.now(tz="UTC")
    rows = forecast_to_archive_rows(forecast, issued_at)
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["issued_at", "anchor", "timestamp"])
        rows = pd.concat([existing, rows], ignore_index=True)
    rows = dedupe(rows)
    rows.to_csv(path, index=False)
    return rows


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def score(archive: pd.DataFrame, ground: pd.DataFrame) -> dict:
    """Join forecasts to what was later measured and compute live skill."""
    g = ground.rename(columns={"pm2_5": "observed"})[["timestamp", "observed"]]
    g["timestamp"] = pd.to_datetime(g["timestamp"])

    a = archive.copy()
    for col in ["issued_at", "anchor", "timestamp"]:
        a[col] = pd.to_datetime(a[col])

    scored = a.merge(g, on="timestamp", how="inner")
    anchor_obs = g.rename(columns={"timestamp": "anchor", "observed": "persistence"})
    scored = scored.merge(anchor_obs, on="anchor", how="inner")
    scored = scored.dropna(subset=["observed", "persistence", "pm2_5_predicted"])

    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "what": "published forecasts scored against later OpenAQ measurements",
        "n_forecasts_issued": int(a["anchor"].nunique()),
        "n_forecasts_scored": int(scored["anchor"].nunique()),
        "n_points": int(len(scored)),
        "buckets": {},
    }
    if scored.empty:
        return out

    # How stale is a forecast by the time it is published? The anchor is the
    # last measured hour it saw; the commit time is when users could see it.
    latency = (a.groupby("anchor")["issued_at"].min().reset_index())
    latency_h = (latency["issued_at"] - latency["anchor"]).dt.total_seconds() / 3600
    out["median_publish_latency_hours"] = round(float(latency_h.median()), 1)

    out["period"] = f"{scored['timestamp'].min():%Y-%m-%d} to {scored['timestamp'].max():%Y-%m-%d}"
    out["overall"] = _summary(scored)

    for lo, hi in BUCKETS:
        part = scored[scored["horizon_hour"].between(lo, hi)]
        if len(part) >= 20:
            out["buckets"][f"{lo}-{hi}h"] = _summary(part)

    has_interval = scored["pm2_5_p10"].notna() & scored["pm2_5_p90"].notna()
    if has_interval.sum() >= 20:
        s = scored[has_interval]
        inside = s["observed"].between(s["pm2_5_p10"], s["pm2_5_p90"])
        out["interval_80"] = {
            "n_points": int(len(s)),
            "coverage": round(float(inside.mean()), 3),
            "mean_width": round(float((s["pm2_5_p90"] - s["pm2_5_p10"]).mean()), 2),
        }
    return out


def _summary(part: pd.DataFrame) -> dict:
    mae = _mae(part["pm2_5_predicted"], part["observed"])
    cams = _mae(part["cams_pm2_5"], part["observed"])
    pers = _mae(part["persistence"], part["observed"])
    return {
        "n_points": int(len(part)),
        "MAE": round(mae, 3),
        "cams_MAE": round(cams, 3),
        "persistence_MAE": round(pers, 3),
        "skill_vs_cams": round(1 - mae / cams, 3) if cams > 0 else None,
        "skill_vs_persistence": round(1 - mae / pers, 3) if pers > 0 else None,
        "bias": round(float((part["pm2_5_predicted"] - part["observed"]).mean()), 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--backfill", action="store_true",
                        help="rebuild the archive from git history first")
    args = parser.parse_args(argv)

    os.makedirs("models", exist_ok=True)

    if args.backfill:
        archive = backfill_from_git()
        if os.path.exists(ARCHIVE_PATH):
            existing = pd.read_csv(ARCHIVE_PATH, parse_dates=["issued_at", "anchor", "timestamp"])
            archive = dedupe(pd.concat([existing, archive], ignore_index=True))
        archive.to_csv(ARCHIVE_PATH, index=False)
        print(f"Archive rebuilt: {archive['anchor'].nunique()} forecasts, {len(archive)} rows")

    if not os.path.exists(ARCHIVE_PATH):
        sys.exit(f"{ARCHIVE_PATH} missing. Run with --backfill first.")

    archive = pd.read_csv(ARCHIVE_PATH, parse_dates=["issued_at", "anchor", "timestamp"])
    ground = pd.read_csv(GROUND_PATH, parse_dates=["timestamp"])
    result = score(archive, ground)

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scored {result['n_forecasts_scored']} of {result['n_forecasts_issued']} "
          f"published forecasts ({result['n_points']} hourly points)")
    if "overall" in result:
        o = result["overall"]
        print(f"  live MAE {o['MAE']:.2f}  CAMS {o['cams_MAE']:.2f}  "
              f"persistence {o['persistence_MAE']:.2f}")
        print(f"  skill vs CAMS {o['skill_vs_cams']:+.1%}  "
              f"vs persistence {o['skill_vs_persistence']:+.1%}  bias {o['bias']:+.2f}")
        print(f"  median publish latency {result['median_publish_latency_hours']} h")
        for name, b in result["buckets"].items():
            print(f"  {name:>7}: MAE {b['MAE']:.2f}  vs CAMS {b['skill_vs_cams']:+.1%}  "
                  f"vs persistence {b['skill_vs_persistence']:+.1%}  (n={b['n_points']})")
    return result


if __name__ == "__main__":
    main()

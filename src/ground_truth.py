"""
Real ground-sensor PM2.5 for Karachi, via OpenAQ.

Why this exists
---------------
The project's training data is CAMS model output — a physics simulation, not a
measurement. That is a legitimate basis for forecasting, but it means the model
learns to predict *what CAMS will say*, not what the air over Karachi actually
contains. The original ground-truth source (the US Consulate WAQI feed) stopped
reporting in March 2025, which left nothing to check CAMS against.

OpenAQ aggregates readings from physical monitors, and Karachi turns out to have
around twenty reporting within the hour. That gives the project two things it
did not have:

  * a genuine live reading for the dashboard, replacing the dead feed
  * a way to measure how far CAMS drifts from real sensors, which is the honest
    answer to "is your model predicting reality or predicting a model?"

A caveat that belongs in the open
---------------------------------
Most of these monitors are low-cost optical sensors (they report pm1 and
um003 particle counts, which is the signature of that hardware). They are known
to over-read PM2.5 in humid conditions and individual units drift. This module
therefore never trusts a single station: it takes the **median** across all
live stations for any given hour, which is robust to a handful of bad units,
and records how many stations contributed so the reading can be discounted when
coverage is thin.

Usage
-----
    python src/ground_truth.py --probe            what history is available
    python src/ground_truth.py --current          latest city reading
    python src/ground_truth.py --history 60       last 60 days, hourly

Needs a free key from https://explore.openaq.org/register in .env:
    OPENAQ_API_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE = "https://api.openaq.org/v3"
LAT, LON = 24.8607, 67.0011
RADIUS_M = 25000
PARAMETER = "pm25"

# A station must have reported this recently to count towards a live reading.
LIVE_MAX_AGE_HOURS = 12

# A sensor needs roughly a year of history to be usable as a training target.
MIN_HISTORY_DAYS = 300

# OpenAQ's free tier allows 60 requests/minute; stay comfortably under it.
REQUEST_PAUSE_S = 1.1
PAGE_LIMIT = 1000


def _key():
    key = os.getenv("OPENAQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENAQ_API_KEY not set.\n"
            "Get a free key at https://explore.openaq.org/register and add to .env:\n"
            "    OPENAQ_API_KEY=your_key_here"
        )
    return key


def _get(path, **params):
    response = requests.get(
        f"{BASE}/{path}",
        headers={"X-API-Key": _key()},
        params=params,
        timeout=45,
    )
    if response.status_code == 401:
        raise SystemExit("OpenAQ rejected the API key. Check OPENAQ_API_KEY in .env")
    if response.status_code == 429:
        print("  rate limited, waiting 30s...")
        time.sleep(30)
        return _get(path, **params)
    response.raise_for_status()
    return response.json()


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def find_pm25_sensors():
    """Every PM2.5 sensor on a monitor within range of Karachi.

    Returns a DataFrame with one row per sensor: sensor id, station name, and
    the window of data that sensor claims to hold.
    """
    payload = _get(
        "locations",
        coordinates=f"{LAT},{LON}",
        radius=RADIUS_M,
        limit=100,
    )
    rows = []
    for loc in payload.get("results", []):
        for sensor in loc.get("sensors", []):
            name = (sensor.get("parameter") or {}).get("name", "")
            if name != PARAMETER:
                continue
            rows.append(
                {
                    "sensor_id": sensor.get("id"),
                    "station": loc.get("name", "?"),
                    "location_id": loc.get("id"),
                    "first": _parse_ts((loc.get("datetimeFirst") or {}).get("utc")),
                    "last": _parse_ts((loc.get("datetimeLast") or {}).get("utc")),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    now = datetime.now(timezone.utc)
    df["age_hours"] = df["last"].map(
        lambda t: (now - t).total_seconds() / 3600 if t is not None else None
    )
    df["history_days"] = [
        (r["last"] - r["first"]).total_seconds() / 86400
        if r["first"] is not None and r["last"] is not None
        else None
        for _, r in df.iterrows()
    ]
    df["live"] = df["age_hours"].fillna(1e9) <= LIVE_MAX_AGE_HOURS
    return df.sort_values("age_hours")


def _hourly_path(sensor_id):
    """Resolve which spelling of the hourly endpoint this API build uses.

    The OpenAQ docs refer to it as both `/hours` and `/measurements_hourly`
    depending on which page you read, so probe once and remember the answer.
    """
    if _hourly_path.resolved:
        return _hourly_path.resolved.format(sensor_id=sensor_id)

    for template in ("sensors/{sensor_id}/hours",
                     "sensors/{sensor_id}/measurements_hourly"):
        try:
            _get(template.format(sensor_id=sensor_id), limit=1)
            _hourly_path.resolved = template
            return template.format(sensor_id=sensor_id)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                continue
            raise
    raise RuntimeError("Neither hourly endpoint responded; check the OpenAQ API docs.")


_hourly_path.resolved = None


def _extract_timestamp(item):
    """Hourly records label their period slightly differently across builds."""
    period = item.get("period") or {}
    for candidate in (
        (period.get("datetimeFrom") or {}).get("utc"),
        (period.get("datetimeTo") or {}).get("utc"),
        (item.get("datetime") or {}).get("utc"),
    ):
        stamp = _parse_ts(candidate)
        if stamp is not None:
            return stamp
    return None


def fetch_sensor_hours(sensor_id, date_from, date_to):
    """All hourly records for one sensor, following pagination to the end.

    A single page caps at 1000 rows — roughly 42 days — so a year of history
    needs about ten pages. Fetching only the first page would silently return a
    tenth of the data and look like it worked.
    """
    path = _hourly_path(sensor_id)
    records = []
    page = 1

    while True:
        payload = _get(
            path,
            datetime_from=date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            datetime_to=date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=PAGE_LIMIT,
            page=page,
        )
        results = payload.get("results", [])
        for item in results:
            stamp = _extract_timestamp(item)
            value = item.get("value")
            if stamp is not None and value is not None:
                records.append({"timestamp": stamp, "pm2_5": float(value)})

        if len(results) < PAGE_LIMIT:
            break
        page += 1
        if page > 40:  # ~4.5 years; a runaway guard, not an expected limit
            print(f"    stopping at page {page} for sensor {sensor_id}")
            break
        time.sleep(REQUEST_PAUSE_S)

    return records


def hourly_history(days=400, sensors=None, min_history_days=MIN_HISTORY_DAYS,
                   use_cache=True):
    """City-level hourly PM2.5: the median across all contributing stations.

    Each sensor's raw pull is cached under data/cache/openaq/. A year of data
    across a dozen sensors is a few hundred API calls and several minutes; the
    cache means a dropped connection costs one sensor, not the whole run.
    """
    if sensors is None:
        sensors = find_pm25_sensors()
    if sensors.empty:
        print("No PM2.5 sensors found.")
        return pd.DataFrame()

    chosen = sensors[sensors["live"]]
    if min_history_days:
        deep = chosen[chosen["history_days"].fillna(0) >= min_history_days]
        if deep.empty:
            print(
                f"No live sensor has {min_history_days}+ days of history; "
                "using all live sensors instead."
            )
        else:
            chosen = deep

    if chosen.empty:
        print("No live PM2.5 sensors found.")
        return pd.DataFrame()

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=days)
    cache_dir = os.path.join("data", "cache", "openaq")
    os.makedirs(cache_dir, exist_ok=True)

    frames = []
    print(f"Pulling up to {days} days of hourly data from {len(chosen)} sensor(s).")
    print("This takes a few minutes. Interrupting is safe - progress is cached.\n")

    for n, (_, row) in enumerate(chosen.iterrows(), start=1):
        sensor_id = int(row["sensor_id"])
        cache_file = os.path.join(cache_dir, f"sensor_{sensor_id}_{days}d.csv")

        if use_cache and os.path.exists(cache_file):
            frame = pd.read_csv(cache_file, parse_dates=["timestamp"])
            print(f"  [{n}/{len(chosen)}] {sensor_id:>9}  {len(frame):>5} hours (cached)")
        else:
            try:
                records = fetch_sensor_hours(sensor_id, date_from, date_to)
            except Exception as exc:
                print(f"  [{n}/{len(chosen)}] {sensor_id:>9}  FAILED: {exc}")
                continue
            if not records:
                print(f"  [{n}/{len(chosen)}] {sensor_id:>9}      0 hours")
                continue
            frame = pd.DataFrame(records)
            frame.to_csv(cache_file, index=False)
            print(
                f"  [{n}/{len(chosen)}] {sensor_id:>9}  {len(frame):>5} hours  "
                f"{row['station'][:36]}"
            )

        frame["sensor_id"] = sensor_id
        frames.append(frame)
        time.sleep(REQUEST_PAUSE_S)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)

    # Discard physically implausible readings before aggregating. Low-cost
    # sensors occasionally emit zeros or spikes in the thousands.
    combined = combined[(combined["pm2_5"] > 0) & (combined["pm2_5"] < 1000)]

    city = (
        combined.groupby("timestamp")
        .agg(pm2_5=("pm2_5", "median"), n_stations=("sensor_id", "nunique"))
        .reset_index()
        .sort_values("timestamp")
    )
    # Convert to Karachi local time to match the rest of the project.
    city["timestamp"] = (
        city["timestamp"].dt.tz_convert("Asia/Karachi").dt.tz_localize(None)
    )
    return city


def current_reading(sensors=None):
    """Latest city-wide PM2.5: median of the freshest value from each station."""
    if sensors is None:
        sensors = find_pm25_sensors()
    live = sensors[sensors["live"]] if not sensors.empty else sensors
    if live.empty:
        return None

    values, contributing, newest = [], [], None
    for location_id in live["location_id"].dropna().unique():
        try:
            payload = _get(f"locations/{int(location_id)}/latest")
        except Exception:
            continue
        for item in payload.get("results", []):
            value = item.get("value")
            stamp = _parse_ts((item.get("datetime") or {}).get("utc"))
            if value is None or stamp is None:
                continue
            if (datetime.now(timezone.utc) - stamp).total_seconds() / 3600 > LIVE_MAX_AGE_HOURS:
                continue
            if 0 < float(value) < 1000:
                values.append(float(value))
                contributing.append(int(location_id))
                newest = stamp if newest is None or stamp > newest else newest
        time.sleep(REQUEST_PAUSE_S)

    if not values:
        return None

    series = pd.Series(values)
    return {
        "timestamp": newest.astimezone(timezone.utc).replace(tzinfo=None)
        + timedelta(hours=5),  # Asia/Karachi
        "pm25_ground": round(float(series.median()), 2),
        "pm25_spread": round(float(series.quantile(0.75) - series.quantile(0.25)), 2),
        "n_stations": len(set(contributing)),
        "source": "OpenAQ / Karachi community monitors (median)",
    }


def update_history(path="data/ground_history.csv", days=10):
    """Append recent measurements to the stored history.

    The daily retrain calls this. Refetching a year every night would be
    wasteful and would hammer a free API, so only the last few days are pulled
    and merged; the overlap absorbs any hours that were still filling in when
    the previous run happened.
    """
    fresh = hourly_history(days=days, min_history_days=None, use_cache=False)
    if fresh.empty:
        print("No new ground data retrieved; keeping existing history.")
        return None

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        before = len(existing)
        combined = pd.concat([existing, fresh], ignore_index=True)
        # Later rows win: a re-fetched hour may now have more stations behind it.
        combined = combined.drop_duplicates(subset="timestamp", keep="last")
    else:
        before = 0
        combined = fresh

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"\nHistory: {before} -> {len(combined)} hourly rows "
          f"({len(combined) - before:+d})")
    print(f"  {combined['timestamp'].min()} -> {combined['timestamp'].max()}")
    return combined


def probe():
    sensors = find_pm25_sensors()
    if sensors.empty:
        print("No PM2.5 sensors found near Karachi.")
        return sensors

    live = sensors[sensors["live"]]
    print(f"PM2.5 sensors within {RADIUS_M / 1000:.0f} km: {len(sensors)}")
    print(f"Reporting in the last {LIVE_MAX_AGE_HOURS}h: {len(live)}\n")

    print(f"  {'sensor':>8}  {'age':>9}  {'history':>11}  station")
    print("  " + "-" * 74)
    for _, r in live.iterrows():
        age = f"{r['age_hours']:.0f}h" if pd.notna(r["age_hours"]) else "?"
        hist = f"{r['history_days']:.0f} days" if pd.notna(r["history_days"]) else "?"
        print(f"  {r['sensor_id']:>8}  {age:>9}  {hist:>11}  {r['station'][:44]}")

    if not live.empty and live["history_days"].notna().any():
        deepest = live["history_days"].max()
        print(f"\nDeepest history among live sensors: {deepest:.0f} days")
        if deepest >= 300:
            print("That is enough to retrain the model on ground truth instead of CAMS.")
        elif deepest >= 60:
            print("Enough to validate CAMS against real sensors, but not to retrain on.")
        else:
            print("Thin history - usable for the live reading only.")
    return sensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="report sensor coverage")
    parser.add_argument("--current", action="store_true", help="fetch latest reading")
    parser.add_argument("--history", type=int, metavar="DAYS", help="fetch hourly history")
    parser.add_argument("--all-sensors", action="store_true",
                        help="include short-history sensors, not just the deep ones")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached pulls")
    parser.add_argument("--update", action="store_true",
                        help="append the last few days to data/ground_history.csv")
    args = parser.parse_args()

    if not any([args.probe, args.current, args.history, args.update]):
        args.probe = True

    os.makedirs("data", exist_ok=True)
    sensors = find_pm25_sensors()

    if args.probe:
        probe()

    if args.current:
        reading = current_reading(sensors)
        if reading:
            pd.DataFrame([reading]).to_csv("data/ground_current.csv", index=False)
            print(
                f"\nGround PM2.5: {reading['pm25_ground']} ug/m3 "
                f"(median of {reading['n_stations']} stations, "
                f"IQR {reading['pm25_spread']}) -> data/ground_current.csv"
            )
        else:
            print("\nNo usable live ground reading.")

    if args.update:
        update_history()

    if args.history:
        city = hourly_history(
            args.history,
            sensors,
            min_history_days=None if args.all_sensors else MIN_HISTORY_DAYS,
            use_cache=not args.no_cache,
        )
        if not city.empty:
            city.to_csv("data/ground_history.csv", index=False)
            print(
                f"\n{len(city)} hourly rows, "
                f"{city['timestamp'].min()} -> {city['timestamp'].max()}"
            )
            print(f"median stations per hour: {city['n_stations'].median():.0f}")
            print("-> data/ground_history.csv")
        else:
            print("\nNo hourly history retrieved.")


if __name__ == "__main__":
    sys.exit(main())

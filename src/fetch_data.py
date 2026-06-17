import requests
import pandas as pd
from datetime import datetime, timedelta

# Karachi coordinates
LAT = 24.8607
LON = 67.0011

# ── WAQI token — real ground sensor (US Embassy Karachi) ──────
WAQI_TOKEN = "775df988c95cb825dbdfd638825a71a19d870fbe"

# -------------------------------------------------------------
# 1. WAQI — Live ground sensor reading
# -------------------------------------------------------------
def fetch_waqi_live():
    """
    Fetches real-time PM2.5 from the US Embassy Karachi ground station.
    This is the same source IQAir uses. Far more accurate than model data.
    """
    url = f"https://api.waqi.info/feed/karachi/?token={WAQI_TOKEN}"
    try:
        r = requests.get(url, timeout=15)
        d = r.json()
        if d["status"] != "ok":
            print(f"  WAQI error: {d}")
            return None

        data    = d["data"]
        iaqi    = data["iaqi"]
        station = data["city"]["name"]
        ts_str  = data["time"]["s"]           # e.g. "2025-03-04 16:00:00"
        ts      = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

        result = {
            "timestamp":   ts,
            "station":     station,
            "aqi_live":    int(data["aqi"]),
            "pm25_live":   float(iaqi.get("pm25", {}).get("v", None) or 0),
            "temperature": float(iaqi.get("t",    {}).get("v", None) or 0),
            "humidity":    float(iaqi.get("h",     {}).get("v", None) or 0),
            "pressure":    float(iaqi.get("p",     {}).get("v", None) or 0),
            "wind_speed":  float(iaqi.get("w",     {}).get("v", None) or 0),
            "dew":         float(iaqi.get("dew",   {}).get("v", None) or 0),
            "source":      "WAQI / US Embassy Karachi Ground Station",
        }

        print(f"  Station  : {station}")
        print(f"  Timestamp: {ts}")
        print(f"  AQI      : {result['aqi_live']}")
        print(f"  PM2.5    : {result['pm25_live']} µg/m³")
        print(f"  Temp     : {result['temperature']}°C")
        print(f"  Humidity : {result['humidity']}%")
        print(f"  Wind     : {result['wind_speed']} m/s")
        return result

    except Exception as e:
        print(f"  WAQI fetch failed: {e}")
        return None


# -------------------------------------------------------------
# 2. Open-Meteo — 1-year historical air quality
# -------------------------------------------------------------
def fetch_air_quality(start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": ["pm2_5","pm10","nitrogen_dioxide",
                   "ozone","carbon_monoxide","sulphur_dioxide"],
        "timezone": "Asia/Karachi",
        "start_date": start_date, "end_date": end_date
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "hourly" not in data:
        print(f"  Air quality API error: {data}")
        return pd.DataFrame()
    d = data["hourly"]
    return pd.DataFrame({
        "timestamp": d["time"],
        "pm2_5": d["pm2_5"],
        "pm10":  d["pm10"],
        "no2":   d["nitrogen_dioxide"],
        "ozone": d["ozone"],
        "co":    d["carbon_monoxide"],
        "so2":   d["sulphur_dioxide"],
    })


# -------------------------------------------------------------
# 3. Open-Meteo — 1-year historical weather
# -------------------------------------------------------------
def fetch_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": ["temperature_2m","relative_humidity_2m",
                   "wind_speed_10m","wind_direction_10m","surface_pressure"],
        "timezone": "Asia/Karachi",
        "start_date": start_date, "end_date": end_date
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "hourly" not in data:
        print(f"  Weather API error: {data}")
        return pd.DataFrame()
    d = data["hourly"]
    return pd.DataFrame({
        "timestamp":   d["time"],
        "temperature": d["temperature_2m"],
        "humidity":    d["relative_humidity_2m"],
        "wind_speed":  d["wind_speed_10m"],
        "wind_dir":    d["wind_direction_10m"],
        "pressure":    d["surface_pressure"],
    })


# -------------------------------------------------------------
# 4. Open-Meteo — 72-hour forecast
# -------------------------------------------------------------
def fetch_forecast():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": ["pm2_5","pm10","nitrogen_dioxide",
                   "ozone","carbon_monoxide","sulphur_dioxide"],
        "timezone": "Asia/Karachi",
        "forecast_days": 3
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "hourly" not in data:
        print(f"  Forecast API error: {data}")
        return pd.DataFrame()
    d = data["hourly"]
    df = pd.DataFrame({
        "timestamp": d["time"],
        "pm2_5": d["pm2_5"],
        "pm10":  d["pm10"],
        "no2":   d["nitrogen_dioxide"],
        "ozone": d["ozone"],
        "co":    d["carbon_monoxide"],
        "so2":   d["sulphur_dioxide"],
    })
    df["is_forecast"] = True
    return df


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    today      = datetime.today()
    end_date   = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"Fetching data: {start_date} → {end_date}\n")

    # ── Step 1: Live ground sensor reading ───────────────────
    print("[1/4] WAQI — Live ground sensor (US Embassy Karachi)...")
    waqi = fetch_waqi_live()

    # ── Step 2: Historical air quality ────────────────────────
    print("\n[2/4] Open-Meteo — Historical air quality (1 year)...")
    aq_df = fetch_air_quality(start_date, end_date)
    print(f"  Rows: {len(aq_df)}")

    # ── Step 3: Historical weather ────────────────────────────
    print("\n[3/4] Open-Meteo — Historical weather (1 year)...")
    wx_df = fetch_weather(start_date, end_date)
    print(f"  Rows: {len(wx_df)}")

    # ── Step 4: 72hr forecast ─────────────────────────────────
    print("\n[4/4] Open-Meteo — 72-hour forecast...")
    fc_df = fetch_forecast()
    print(f"  Rows: {len(fc_df)}")

    if aq_df.empty or wx_df.empty:
        print("\nMissing critical data. Exiting.")
        return

    # ── Merge historical ──────────────────────────────────────
    aq_df["timestamp"] = pd.to_datetime(aq_df["timestamp"])
    wx_df["timestamp"] = pd.to_datetime(wx_df["timestamp"])
    fc_df["timestamp"] = pd.to_datetime(fc_df["timestamp"])

    historical = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    historical["is_forecast"] = False

    # ── WAQI calibration ──────────────────────────────────────
    # If WAQI live reading is available, we store it separately.
    # The dashboard uses it for the "current AQI" display card.
    # Historical training still uses Open-Meteo (1 year of data).
    if waqi:
        waqi_df = pd.DataFrame([{
            "timestamp":   waqi["timestamp"],
            "aqi_live":    waqi["aqi_live"],
            "pm25_live":   waqi["pm25_live"],
            "temperature": waqi["temperature"],
            "humidity":    waqi["humidity"],
            "pressure":    waqi["pressure"],
            "wind_speed":  waqi["wind_speed"],
            "station":     waqi["station"],
            "source":      waqi["source"],
        }])
        waqi_df.to_csv("data/waqi_live.csv", index=False)
        print(f"\n  WAQI live reading saved → data/waqi_live.csv")

    # ── Forecast weather columns ──────────────────────────────
    for col in ["temperature","humidity","wind_speed","wind_dir","pressure"]:
        if col not in fc_df.columns:
            fc_df[col] = None

    # ── Combine & save ────────────────────────────────────────
    full_df = pd.concat([historical, fc_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset="timestamp")
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    full_df.to_csv("data/raw_data.csv", index=False)
    historical.to_csv("data/raw_air_quality.csv", index=False)

    print(f"\nDone!")
    print(f"  Historical rows : {len(historical)}")
    print(f"  Forecast rows   : {len(fc_df)}")
    print(f"  Total saved     : {len(full_df)}")
    if waqi:
        print(f"\nLive AQI (ground sensor): {waqi['aqi_live']}")
        print(f"Live PM2.5 (ground sensor): {waqi['pm25_live']} µg/m³")
        print(f"Source: {waqi['source']}")


if __name__ == "__main__":
    main()
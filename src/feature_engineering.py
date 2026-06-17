import pandas as pd
import numpy as np

# ----------------------------------
# EPA Official AQI Formula
# ----------------------------------
def calc_aqi_pm25(pm25):
    # Updated EPA 2024 breakpoints (effective May 6, 2024)
    breakpoints = [
        (0.0,   9.0,   0,  50),
        (9.1,  35.4,  51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5,225.4, 201, 300),
        (225.5, 500,  301, 500),
    ]
    for lo, hi, alo, ahi in breakpoints:
        if lo <= pm25 <= hi:
            return round(((ahi - alo) / (hi - lo)) * (pm25 - lo) + alo)
    return 500

def calc_aqi_pm10(pm10):
    breakpoints = [
        (0,   54,   0,  50),
        (55,  154,  51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ]
    for lo, hi, alo, ahi in breakpoints:
        if lo <= pm10 <= hi:
            return round(((ahi - alo) / (hi - lo)) * (pm10 - lo) + alo)
    return 500

def calc_aqi_no2(no2_ppb):
    breakpoints = [
        (0,    53,   0,  50),
        (54,  100,  51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650,1249, 201, 300),
        (1250,2049,301, 500),
    ]
    for lo, hi, alo, ahi in breakpoints:
        if lo <= no2_ppb <= hi:
            return round(((ahi - alo) / (hi - lo)) * (no2_ppb - lo) + alo)
    return 500

def compute_aqi(row):
    # NO2: ug/m3 to ppb (divide by 1.88)
    no2_ppb = row["no2"] / 1.88 if pd.notna(row["no2"]) else 0
    sub_indices = [
        calc_aqi_pm25(row["pm2_5"]) if pd.notna(row["pm2_5"]) else 0,
        calc_aqi_pm10(row["pm10"])  if pd.notna(row["pm10"])  else 0,
        calc_aqi_no2(no2_ppb),
    ]
    return max(sub_indices)  # EPA: final AQI = max sub-index

# ----------------------------------
# Main Feature Engineering
# ----------------------------------
def engineer_features(df):
    df = df.copy()

    # Sirf historical data use karo training ke liye
    if "is_forecast" in df.columns:
        df = df[df["is_forecast"] == False].copy()

    # 1. Missing values
    poll_cols = ["pm2_5", "pm10", "no2", "ozone", "co", "so2"]
    wx_cols   = ["temperature", "humidity", "wind_speed", "wind_dir", "pressure"]
    all_cols  = poll_cols + wx_cols

    for col in all_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].fillna(df[col].median())

    # 2. Percentile capping (1st-99th)
    for col in all_cols:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    # 3. EPA AQI calculate karo
    df["aqi"] = df.apply(compute_aqi, axis=1)

    # 4. Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"]= df["hour"].isin([7,8,9,17,18,19]).astype(int)

    # 5. Cyclic encoding (Mariam se better)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 6. Lag features
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"pm2_5_lag{lag}"]  = df["pm2_5"].shift(lag)
        df[f"aqi_lag{lag}"]    = df["aqi"].shift(lag)

    # 7. Rolling averages
    for w in [3, 6, 12, 24]:
        df[f"pm2_5_roll{w}"] = df["pm2_5"].rolling(window=w).mean()
        df[f"aqi_roll{w}"]   = df["aqi"].rolling(window=w).mean()

    # 8. Interaction features
    df["pm_ratio"]       = df["pm2_5"] / (df["pm10"] + 1e-6)
    df["wind_dispersion"]= df["wind_speed"] / (df["pm2_5"] + 1e-6)
    df["heat_humidity"]  = df["temperature"] * df["humidity"] / 100

    # 9. Drop NaN rows (lag se bani hain)
    df = df.dropna().reset_index(drop=True)

    df.to_csv("data/featured_data.csv", index=False)
    print(f"Features ready! Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nAQI Stats:")
    print(f"  Min  : {df['aqi'].min():.1f}")
    print(f"  Max  : {df['aqi'].max():.1f}")
    print(f"  Mean : {df['aqi'].mean():.1f}")
    return df

if __name__ == "__main__":
    raw = pd.read_csv(
        "data/raw_air_quality.csv",
        parse_dates=["timestamp"]
    )
    engineer_features(raw)
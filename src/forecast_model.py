import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

FEATURE_COLS = [
    'no2', 'ozone', 'co', 'so2', 'pm10',
    'temperature', 'humidity', 'wind_speed', 'wind_dir', 'pressure',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'pm2_5_lag1', 'aqi_lag1', 'pm2_5_lag3', 'aqi_lag3',
    'pm2_5_lag6', 'aqi_lag6', 'pm2_5_lag12', 'aqi_lag12',
    'pm2_5_lag24', 'aqi_lag24', 'pm2_5_lag48', 'aqi_lag48',
    'pm2_5_roll3', 'aqi_roll3', 'pm2_5_roll6', 'aqi_roll6',
    'pm2_5_roll12', 'aqi_roll12', 'pm2_5_roll24', 'aqi_roll24',
    'pm_ratio', 'wind_dispersion', 'heat_humidity'
]

def calc_aqi_pm25(pm25):
    breakpoints = [
        (0.0,  12.0,   0,  50),
        (12.1, 35.4,  51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5,250.4, 201, 300),
        (250.5,500.4, 301, 500),
    ]
    for lo, hi, alo, ahi in breakpoints:
        if lo <= pm25 <= hi:
            return float(round(((ahi-alo)/(hi-lo))*(pm25-lo)+alo))
    return 500.0

def recursive_forecast(model, last_known_row, steps=72):
    """
    Recursively predict future PM2.5 values.
    Each prediction becomes input for the next step.
    """
    history = last_known_row.copy()
    predictions = []

    for step in range(1, steps + 1):
        # Build feature row
        row = {}

        # Static/slowly changing features — carry forward
        for col in ['no2', 'ozone', 'co', 'so2', 'pm10',
                    'temperature', 'humidity', 'wind_speed',
                    'wind_dir', 'pressure']:
            row[col] = history.get(col, 0)

        # Time features for next hour
        next_hour       = int((history['hour'] + step) % 24)
        next_dow        = int((history['day_of_week'] + step // 24) % 7)
        next_month      = int(history['month'])
        row['hour']         = next_hour
        row['day_of_week']  = next_dow
        row['month']        = next_month
        row['is_weekend']   = 1 if next_dow in [5, 6] else 0
        row['is_rush_hour'] = 1 if next_hour in [7,8,9,17,18,19] else 0

        # Cyclic encoding
        row['hour_sin']  = np.sin(2*np.pi*next_hour/24)
        row['hour_cos']  = np.cos(2*np.pi*next_hour/24)
        row['month_sin'] = np.sin(2*np.pi*next_month/12)
        row['month_cos'] = np.cos(2*np.pi*next_month/12)
        row['dow_sin']   = np.sin(2*np.pi*next_dow/7)
        row['dow_cos']   = np.cos(2*np.pi*next_dow/7)

        # Lag features from prediction history
        all_preds = [history['pm2_5']] + predictions
        def get_lag(n):
            if len(all_preds) >= n:
                return all_preds[-n]
            return history.get(f'pm2_5_lag{n}', all_preds[0])

        row['pm2_5_lag1']  = get_lag(1)
        row['pm2_5_lag3']  = get_lag(3)
        row['pm2_5_lag6']  = get_lag(6)
        row['pm2_5_lag12'] = get_lag(12)
        row['pm2_5_lag24'] = get_lag(24)
        row['pm2_5_lag48'] = get_lag(48)

        # AQI lags
        row['aqi_lag1']  = calc_aqi_pm25(get_lag(1))
        row['aqi_lag3']  = calc_aqi_pm25(get_lag(3))
        row['aqi_lag6']  = calc_aqi_pm25(get_lag(6))
        row['aqi_lag12'] = calc_aqi_pm25(get_lag(12))
        row['aqi_lag24'] = calc_aqi_pm25(get_lag(24))
        row['aqi_lag48'] = calc_aqi_pm25(get_lag(48))

        # Rolling averages
        window = [history['pm2_5']] + predictions
        row['pm2_5_roll3']  = np.mean(window[-3:]  if len(window)>=3  else window)
        row['pm2_5_roll6']  = np.mean(window[-6:]  if len(window)>=6  else window)
        row['pm2_5_roll12'] = np.mean(window[-12:] if len(window)>=12 else window)
        row['pm2_5_roll24'] = np.mean(window[-24:] if len(window)>=24 else window)

        row['aqi_roll3']  = calc_aqi_pm25(row['pm2_5_roll3'])
        row['aqi_roll6']  = calc_aqi_pm25(row['pm2_5_roll6'])
        row['aqi_roll12'] = calc_aqi_pm25(row['pm2_5_roll12'])
        row['aqi_roll24'] = calc_aqi_pm25(row['pm2_5_roll24'])

        # Interaction features
        pm10 = row['pm10'] if row['pm10'] > 0 else 1
        row['pm_ratio']        = row['pm2_5_lag1'] / (pm10 + 1e-6)
        row['wind_dispersion'] = row['wind_speed'] / (row['pm2_5_lag1'] + 1e-6)
        row['heat_humidity']   = row['temperature'] * row['humidity'] / 100

        # Predict
        X_row = pd.DataFrame([row])[FEATURE_COLS]
        pred  = float(model.predict(X_row)[0])
        pred  = max(0, pred)  # PM2.5 cannot be negative
        predictions.append(pred)

    return predictions

def evaluate_recursive(model, df, horizon_hours):
    """Evaluate recursive forecasting accuracy on test set."""
    test_start = int(len(df) * 0.8)
    # Sample 50 starting points from test set
    test_indices = range(test_start, len(df) - horizon_hours, 10)
    test_indices = list(test_indices)[:50]

    maes = []
    for idx in test_indices:
        last_row  = df.iloc[idx].to_dict()
        actuals   = df['pm2_5'].iloc[idx+1 : idx+1+horizon_hours].values
        preds     = recursive_forecast(model, last_row, steps=horizon_hours)
        preds     = preds[:len(actuals)]
        if len(actuals) == len(preds):
            maes.append(mean_absolute_error(actuals, preds))

    return np.mean(maes) if maes else None

def main():
    df = pd.read_csv("data/featured_data.csv", parse_dates=["timestamp"])

    # Train base model
    X = df[FEATURE_COLS]
    y = df["pm2_5"]
    split = int(len(df) * 0.8)

    print("Training base XGBoost model for recursive forecasting...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X.iloc[:split], y.iloc[:split])

    print("\nEvaluating recursive multi-step accuracy...")
    results = {}
    for label, hours in [("24hr", 24), ("48hr", 48), ("72hr", 72)]:
        mae = evaluate_recursive(model, df, hours)
        results[label] = {"MAE": round(mae, 3), "horizon_hours": hours}
        print(f"  {label}  ->  MAE: {mae:.3f} ug/m3")

    print("\n" + "="*40)
    print("Recursive Forecast Summary:")
    print("="*40)
    for label, res in results.items():
        print(f"  {label}  ->  MAE: {res['MAE']}")
    print("="*40)

    # Generate actual 72hr forecast from latest data
    print("\nGenerating live 72-hour forecast...")
    last_row   = df.iloc[-1].to_dict()
    forecast_72 = recursive_forecast(model, last_row, steps=72)

    last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    timestamps = [last_ts + pd.Timedelta(hours=i+1) for i in range(72)]

    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "pm2_5_predicted": [round(p, 2) for p in forecast_72],
        "aqi_predicted": [calc_aqi_pm25(p) for p in forecast_72],
        "horizon_hour": list(range(1, 73))
    })

    # AQI category labels
    def aqi_label(aqi):
        if aqi <= 50:   return "Good"
        if aqi <= 100:  return "Moderate"
        if aqi <= 150:  return "Unhealthy for Sensitive Groups"
        if aqi <= 200:  return "Unhealthy"
        if aqi <= 300:  return "Very Unhealthy"
        return "Hazardous"

    forecast_df["aqi_category"] = forecast_df["aqi_predicted"].apply(aqi_label)

    forecast_df.to_csv("data/forecast_72hr.csv", index=False)

    print("\nSample 72-hour forecast:")
    print(forecast_df[["timestamp","pm2_5_predicted",
                        "aqi_predicted","aqi_category"]].iloc[::12].to_string(index=False))

    with open("models/horizon_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("models/recursive_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nForecast saved to data/forecast_72hr.csv")

if __name__ == "__main__":
    main()
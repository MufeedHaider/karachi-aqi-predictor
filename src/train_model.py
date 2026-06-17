import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import json

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

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  {name}:")
    print(f"    MAE  : {mae:.3f}  (avg error in ug/m3)")
    print(f"    RMSE : {rmse:.3f}")
    print(f"    R2   : {r2:.3f}  (1.0 = perfect)")
    return {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,3)}

def train_and_evaluate():
    df = pd.read_csv("data/featured_data.csv", parse_dates=["timestamp"])

    X = df[FEATURE_COLS]
    y = df["pm2_5"]

    # Time Series Split — shuffle=False zaroori hai
    # Last 20% test, baaki train
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Features: {len(FEATURE_COLS)}")
    print("\nTraining models...")

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest"   : RandomForestRegressor(
                                n_estimators=200,
                                max_depth=15,
                                min_samples_leaf=5,
                                random_state=42,
                                n_jobs=-1
                            ),
        "XGBoost"         : xgb.XGBRegressor(
                                n_estimators=300,
                                learning_rate=0.05,
                                max_depth=6,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                verbosity=0
                            ),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = evaluate(y_test, preds, name)
        trained[name] = model

    # Best model = lowest MAE
    best_name = min(results, key=lambda x: results[x]["MAE"])
    best_model = trained[best_name]

    print(f"\n{'='*40}")
    print(f"Best Model: {best_name}")
    print(f"MAE  : {results[best_name]['MAE']}")
    print(f"RMSE : {results[best_name]['RMSE']}")
    print(f"R2   : {results[best_name]['R2']}")
    print(f"{'='*40}")

    # Save karo
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    with open("models/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    print("\nAll models saved!")
    return results, best_name

if __name__ == "__main__":
    train_and_evaluate()
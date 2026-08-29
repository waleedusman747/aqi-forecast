import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"
MODEL_DIR = "models"

BASE_FEATURE_COLS = ["pm25", "pm10", "o3", "no2", "so2", "co",
                      "temperature", "humidity", "pressure", "wind",
                      "hour", "day_of_week", "month", "is_weekend"]

LAG_FEATURE_COLS = ["aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h",
                     "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
                     "aqi_rolling_mean_24h", "aqi_change_rate_1h"]

FEATURE_COLS = BASE_FEATURE_COLS + LAG_FEATURE_COLS


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_lag_features(df):
    """Add lag and rolling-window features based on AQI history."""
    df = df.copy()

    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    df["aqi_rolling_mean_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_std_6h"] = df["aqi"].rolling(window=6).std()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24).mean()

    df["aqi_change_rate_1h"] = df["aqi"] - df["aqi_lag_1h"]

    return df


def build_targets(df, horizon_hours):
    df = df.copy()
    df[f"aqi_target_{horizon_hours}h"] = df["aqi"].shift(-horizon_hours)
    return df


def prepare_dataset(df, horizon_hours):
    df = build_targets(df, horizon_hours)
    target_col = f"aqi_target_{horizon_hours}h"

    cols_needed = FEATURE_COLS + [target_col]
    df = df.dropna(subset=cols_needed)

    X = df[FEATURE_COLS]
    y = df[target_col]
    return X, y


def time_based_split(X, y, test_fraction=0.2):
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_evaluate(df, horizon_hours):
    print(f"\n{'='*50}")
    print(f"Training models for {horizon_hours}h-ahead AQI forecast")
    print(f"{'='*50}")

    X, y = prepare_dataset(df, horizon_hours)
    X_train, X_test, y_train, y_test = time_based_split(X, y)

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    results = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results["Ridge"] = {"model": ridge, "metrics": evaluate(y_test, ridge_preds)}

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["RandomForest"] = {"model": rf, "metrics": evaluate(y_test, rf_preds)}

    for name, res in results.items():
        m = res["metrics"]
        print(f"{name}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best_model = results[best_name]["model"]
    print(f"Best model for {horizon_hours}h: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f"{MODEL_DIR}/aqi_model_{horizon_hours}h.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved to {model_path}")

    return results


if __name__ == "__main__":
    df = load_data()
    df = add_lag_features(df)

    for horizon in [24, 48, 72]:
        train_and_evaluate(df, horizon)
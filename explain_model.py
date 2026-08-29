import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"

BASE_FEATURE_COLS = ["pm25", "pm10", "o3", "no2", "so2", "co",
                      "temperature", "humidity", "pressure", "wind",
                      "hour", "day_of_week", "month", "is_weekend"]

LAG_FEATURE_COLS = ["aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h",
                     "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
                     "aqi_rolling_mean_24h", "aqi_change_rate_1h"]

FEATURE_COLS = BASE_FEATURE_COLS + LAG_FEATURE_COLS


def add_lag_features(df):
    df = df.copy()
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_rolling_mean_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_std_6h"] = df["aqi"].rolling(window=6).std()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24).mean()
    df["aqi_change_rate_1h"] = df["aqi"] - df["aqi_lag_1h"]
    return df


def explain(horizon_hours):
    print(f"Explaining {horizon_hours}h model...")

    model = joblib.load(f"models/aqi_model_{horizon_hours}h.pkl")

    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_lag_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    # Use a sample of recent rows for the explanation (SHAP can be slow on huge data)
    X_sample = df[FEATURE_COLS].tail(200)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot: which features matter most overall
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"models/shap_summary_{horizon_hours}h.png", dpi=150)
    plt.close()
    print(f"Saved models/shap_summary_{horizon_hours}h.png")

    # Feature importance bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"models/shap_importance_{horizon_hours}h.png", dpi=150)
    plt.close()
    print(f"Saved models/shap_importance_{horizon_hours}h.png")


if __name__ == "__main__":
    for horizon in [24, 48, 72]:
        explain(horizon)
import requests
import pandas as pd
from datetime import datetime, timedelta

CITY = "lahore"
LATITUDE = 31.5497
LONGITUDE = 74.3436
PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


def fetch_historical_air_quality():
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
        f"&start_date={START_DATE}&end_date={END_DATE}&timezone=auto"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def fetch_historical_weather():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
        f"&start_date={START_DATE}&end_date={END_DATE}&timezone=auto"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def build_dataframe(air_data, weather_data):
    air_hourly = air_data["hourly"]
    weather_hourly = weather_data["hourly"]

    df_air = pd.DataFrame({
        "timestamp": pd.to_datetime(air_hourly["time"]),
        "aqi": air_hourly["us_aqi"],
        "pm25": air_hourly["pm2_5"],
        "pm10": air_hourly["pm10"],
        "o3": air_hourly["ozone"],
        "no2": air_hourly["nitrogen_dioxide"],
        "so2": air_hourly["sulphur_dioxide"],
        "co": air_hourly["carbon_monoxide"],
    })

    df_weather = pd.DataFrame({
        "timestamp": pd.to_datetime(weather_hourly["time"]),
        "temperature": weather_hourly["temperature_2m"],
        "humidity": weather_hourly["relative_humidity_2m"],
        "pressure": weather_hourly["surface_pressure"],
        "wind": weather_hourly["wind_speed_10m"],
    })

    df = pd.merge(df_air, df_weather, on="timestamp", how="inner")
    df["city"] = CITY
    df = df.dropna(subset=["aqi"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    numeric_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                     "temperature", "humidity", "pressure", "wind"]
    df[numeric_cols] = df[numeric_cols].astype("float64")

    int_cols = ["hour", "day_of_week", "month", "is_weekend"]
    df[int_cols] = df[int_cols].astype("int64")

    df = df[["city", "timestamp", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
             "temperature", "humidity", "pressure", "wind",
             "hour", "day_of_week", "month", "is_weekend"]]

    return df


if __name__ == "__main__":
    print(f"Fetching data from {START_DATE} to {END_DATE}...")

    print("Fetching historical air quality data...")
    air_data = fetch_historical_air_quality()

    print("Fetching historical weather data...")
    weather_data = fetch_historical_weather()

    print("Building combined dataset...")
    df_backfill = build_dataframe(air_data, weather_data)

    print(f"Backfilled {len(df_backfill)} rows.")
    print(f"Date range in data: {df_backfill['timestamp'].min()} to {df_backfill['timestamp'].max()}")

    df_backfill.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved to {PARQUET_PATH}")
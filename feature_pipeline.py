import requests
import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Load secret keys from .env file
load_dotenv()
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

CITY = "lahore"
PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"


def fetch_aqi_data():
    """Fetch live AQI data from AQICN API."""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        raise Exception("Failed to fetch data from AQICN API")

    return data["data"]


def extract_features(raw_data):
    """Extract and clean the fields we care about."""
    iaqi = raw_data.get("iaqi", {})

    def get_value(key):
        return iaqi.get(key, {}).get("v")

    now = datetime.now()

    row = {
        "city": CITY,
        "timestamp": now,
        "aqi": raw_data.get("aqi"),
        "pm25": get_value("pm25"),
        "pm10": get_value("pm10"),
        "o3": get_value("o3"),
        "no2": get_value("no2"),
        "so2": get_value("so2"),
        "co": get_value("co"),
        "temperature": get_value("t"),
        "humidity": get_value("h"),
        "pressure": get_value("p"),
        "wind": get_value("w"),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "month": now.month,
        "is_weekend": 1 if now.weekday() >= 5 else 0,
    }
    return row


def save_to_parquet(row):
    """Append the new row to our Parquet file (create it if it doesn't exist)."""
    df_new = pd.DataFrame([row])

    numeric_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                     "temperature", "humidity", "pressure", "wind"]
    df_new[numeric_cols] = df_new[numeric_cols].astype("float64")

    int_cols = ["hour", "day_of_week", "month", "is_weekend"]
    df_new[int_cols] = df_new[int_cols].astype("int64")

    if os.path.exists(PARQUET_PATH):
        df_old = pd.read_parquet(PARQUET_PATH)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved. Total rows in file: {len(df_combined)}")


def materialize_to_feast():
    """Push the latest data into Feast's online store."""
    print("Materializing data into Feast online store...")
    subprocess.run(
        ["feast", "materialize-incremental", datetime.now().isoformat()],
        cwd="aqi_repo/feature_repo",
        check=True,
    )
    print("Materialization done.")


if __name__ == "__main__":
    raw = fetch_aqi_data()
    row = extract_features(raw)

    print("Data fetched:")
    print(pd.DataFrame([row]).T)

    save_to_parquet(row)
    materialize_to_feast()
    print("Pipeline complete!")
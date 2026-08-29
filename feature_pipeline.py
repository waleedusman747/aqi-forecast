import requests
import os
import subprocess
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
IQAIR_API_KEY = os.getenv("IQAIR_API_KEY")

CITY = "lahore"
PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"


def fetch_iqair_data():
    """Fetch live AQI + weather data from IQAir's free Community API."""
    url = (
        "https://api.airvisual.com/v2/city"
        f"?city=Lahore&state=Punjab&country=Pakistan&key={IQAIR_API_KEY}"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if data["status"] != "success":
        raise Exception(f"IQAir API error: {data}")

    return data["data"]


def extract_features(raw_data):
    """Convert IQAir's response into our standard feature row."""
    pollution = raw_data["current"]["pollution"]
    weather = raw_data["current"]["weather"]

    ts = pd.to_datetime(pollution["ts"]).tz_convert("Asia/Karachi").tz_localize(None)
    row = {
        "city": CITY,
        "timestamp": ts,
        "aqi": pollution["aqius"],
        # IQAir's free tier doesn't give individual pollutant breakdowns,
        # so we leave these as NaN (handled downstream same as before)
        "pm25": None,
        "pm10": None,
        "o3": None,
        "no2": None,
        "so2": None,
        "co": None,
        "temperature": weather["tp"],
        "humidity": weather["hu"],
        "pressure": weather["pr"],
        "wind": weather["ws"],
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "is_weekend": 1 if ts.weekday() >= 5 else 0,
    }
    return row


def save_to_parquet(row):
    df_new = pd.DataFrame([row])

    numeric_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                     "temperature", "humidity", "pressure", "wind"]
    df_new[numeric_cols] = df_new[numeric_cols].astype("float64")

    int_cols = ["hour", "day_of_week", "month", "is_weekend"]
    df_new[int_cols] = df_new[int_cols].astype("int64")

    col_order = ["city", "timestamp", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                 "temperature", "humidity", "pressure", "wind",
                 "hour", "day_of_week", "month", "is_weekend"]
    df_new = df_new[col_order]

    if os.path.exists(PARQUET_PATH):
        df_old = pd.read_parquet(PARQUET_PATH)
        df_old = df_old[df_old["timestamp"] != row["timestamp"]]
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    df_combined.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved. Total rows in file: {len(df_combined)}")


def materialize_to_feast():
    print("Materializing data into Feast online store...")
    subprocess.run(
        ["feast", "materialize-incremental", datetime.now().isoformat()],
        cwd="aqi_repo/feature_repo",
        check=True,
    )
    print("Materialization done.")


if __name__ == "__main__":
    print("Fetching latest data from IQAir...")
    raw = fetch_iqair_data()
    row = extract_features(raw)

    print("Latest data point:")
    print(pd.DataFrame([row]).T)

    save_to_parquet(row)
    materialize_to_feast()
    print("Pipeline complete!")
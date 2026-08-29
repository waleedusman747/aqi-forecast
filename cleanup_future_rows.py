import pandas as pd

PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"

df = pd.read_parquet(PARQUET_PATH)
now = pd.Timestamp.now()

before = len(df)
df = df[df["timestamp"] <= now]
after = len(df)

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_parquet(PARQUET_PATH, index=False)

print(f"Removed {before - after} future-dated rows.")
print(f"Latest timestamp now: {df['timestamp'].max()}")
print(f"Latest AQI: {df['aqi'].iloc[-1]}")
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# Entity: the "thing" our features are about (here, a city)
city = Entity(name="city", join_keys=["city"])

# Source: where Feast reads the raw data from (a Parquet file we keep updating)
aqi_source = FileSource(
    name="aqi_source",
    path="data/aqi_data.parquet",
    timestamp_field="timestamp",
)

# Feature View: defines the actual features and their types
aqi_features_view = FeatureView(
    name="aqi_features_view",
    entities=[city],
    ttl=timedelta(days=365),
    schema=[
        Field(name="aqi", dtype=Float64),
        Field(name="pm25", dtype=Float64),
        Field(name="pm10", dtype=Float64),
        Field(name="o3", dtype=Float64),
        Field(name="no2", dtype=Float64),
        Field(name="so2", dtype=Float64),
        Field(name="co", dtype=Float64),
        Field(name="temperature", dtype=Float64),
        Field(name="humidity", dtype=Float64),
        Field(name="pressure", dtype=Float64),
        Field(name="wind", dtype=Float64),
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
    ],
    source=aqi_source,
)
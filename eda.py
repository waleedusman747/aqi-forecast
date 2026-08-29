import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"
OUTPUT_DIR = "eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def plot_aqi_over_time(df):
    """Plot 1: How AQI changes over the full collected period."""
    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df["aqi"], linewidth=0.8, color="#ff7e00")
    plt.title("AQI Over Time — Lahore")
    plt.xlabel("Date")
    plt.ylabel("AQI (US)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aqi_over_time.png", dpi=150)
    plt.close()
    print("Saved: aqi_over_time.png")


def plot_hourly_pattern(df):
    """Plot 2: Average AQI by hour of day — is smog worse at certain times?"""
    hourly_avg = df.groupby("hour")["aqi"].mean()
    plt.figure(figsize=(10, 5))
    hourly_avg.plot(kind="bar", color="#4fc3f7")
    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hour (24h format)")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aqi_by_hour.png", dpi=150)
    plt.close()
    print("Saved: aqi_by_hour.png")


def plot_monthly_pattern(df):
    """Plot 3: Average AQI by month — seasonal smog pattern."""
    monthly_avg = df.groupby("month")["aqi"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(10, 5))
    monthly_avg.index = [month_names[m - 1] for m in monthly_avg.index]
    monthly_avg.plot(kind="bar", color="#8f3f97")
    plt.title("Average AQI by Month (Seasonal Pattern)")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aqi_by_month.png", dpi=150)
    plt.close()
    print("Saved: aqi_by_month.png")


def plot_weekday_vs_weekend(df):
    """Plot 4: Does AQI differ on weekends (less traffic) vs weekdays?"""
    comparison = df.groupby("is_weekend")["aqi"].mean()
    comparison.index = ["Weekday", "Weekend"]
    plt.figure(figsize=(6, 5))
    comparison.plot(kind="bar", color=["#4fc3f7", "#ff7e00"])
    plt.title("Average AQI: Weekday vs Weekend")
    plt.ylabel("Average AQI")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/weekday_vs_weekend.png", dpi=150)
    plt.close()
    print("Saved: weekday_vs_weekend.png")


def plot_correlation_heatmap(df):
    """Plot 5: Correlation between AQI and weather/pollutant features."""
    cols = ["aqi", "temperature", "humidity", "pressure", "wind"]
    available_cols = [c for c in cols if c in df.columns and df[c].notna().sum() > 0]
    corr = df[available_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation: AQI vs Weather Features")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: correlation_heatmap.png")


def plot_aqi_distribution(df):
    """Plot 6: Distribution of AQI values — how often is it in each category?"""
    plt.figure(figsize=(10, 5))
    plt.hist(df["aqi"], bins=40, color="#ff7e00", edgecolor="black", alpha=0.8)
    plt.axvline(50, color="green", linestyle="--", label="Good/Moderate boundary")
    plt.axvline(100, color="orange", linestyle="--", label="Moderate/Unhealthy(sensitive) boundary")
    plt.axvline(150, color="red", linestyle="--", label="Unhealthy boundary")
    plt.title("Distribution of AQI Values")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aqi_distribution.png", dpi=150)
    plt.close()
    print("Saved: aqi_distribution.png")


def print_summary_stats(df):
    """Print basic summary statistics to the console."""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(df["aqi"].describe())
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nDays classified as 'Unhealthy or worse' (AQI > 150): "
          f"{(df['aqi'] > 150).sum()} out of {len(df)} readings "
          f"({(df['aqi'] > 150).mean() * 100:.1f}%)")


if __name__ == "__main__":
    df = load_data()

    print_summary_stats(df)

    plot_aqi_over_time(df)
    plot_hourly_pattern(df)
    plot_monthly_pattern(df)
    plot_weekday_vs_weekend(df)
    plot_correlation_heatmap(df)
    plot_aqi_distribution(df)

    print(f"\nAll plots saved to '{OUTPUT_DIR}/' folder.")
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

CITY = "Lahore"
PARQUET_PATH = "aqi_repo/feature_repo/data/aqi_data.parquet"

BASE_FEATURE_COLS = ["pm25", "pm10", "o3", "no2", "so2", "co",
                      "temperature", "humidity", "pressure", "wind",
                      "hour", "day_of_week", "month", "is_weekend"]

LAG_FEATURE_COLS = ["aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h",
                     "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
                     "aqi_rolling_mean_24h", "aqi_change_rate_1h"]

FEATURE_COLS = BASE_FEATURE_COLS + LAG_FEATURE_COLS


def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#dbc400"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


@st.cache_data(ttl=3600)
def load_historical_data():
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


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


@st.cache_resource
def load_models():
    models = {}
    for horizon in [24, 48, 72]:
        models[horizon] = joblib.load(f"models/aqi_model_{horizon}h.pkl")
    return models


def make_predictions(df, models):
    df_with_lags = add_lag_features(df)
    latest_row = df_with_lags.dropna(subset=FEATURE_COLS).iloc[[-1]]
    X_latest = latest_row[FEATURE_COLS]

    predictions = {}
    for horizon, model in models.items():
        predictions[horizon] = model.predict(X_latest)[0]

    return predictions, latest_row


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")

# ================= THEME STATE =================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# Theme palette
if is_dark:
    bg_color = "#0e1117"
    card_bg = "#1c2029"
    text_color = "#f0f2f6"
    subtext_color = "#a0a5b1"
    accent = "#4fc3f7"
    border_color = "#2d3340"
else:
    bg_color = "#f7f9fc"
    card_bg = "#ffffff"
    text_color = "#1a1f2b"
    subtext_color = "#5c6270"
    accent = "#0277bd"
    border_color = "#e0e4ea"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}
[data-testid="stMetricValue"] {{ font-size: 2.2rem; color: {text_color}; }}
[data-testid="stMetricLabel"] {{ color: {subtext_color}; }}
.category-badge {{
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-weight: 600;
    color: black;
    font-size: 0.9rem;
}}
.custom-card {{
    background-color: {card_bg};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}}
.accent-text {{ color: {accent}; }}
h1, h2, h3 {{ color: {text_color} !important; }}
p, span, div {{ color: {text_color}; }}
[data-testid="stSidebar"] {{
    background-color: {card_bg};
    border-right: 1px solid {border_color};
}}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"### 🌫️ AQI Forecast")
    with col_b:
        icon = "🌙" if is_dark else "☀️"
        st.button(icon, on_click=toggle_theme, help="Toggle dark/light mode")

    st.caption("Serverless AQI Forecasting Pipeline")
    st.markdown("---")
    st.markdown("**📍 Location**")
    st.write(f"{CITY}, Pakistan")
    st.markdown("---")
    st.markdown("**⚙️ How it works**")
    st.caption(
        "Hourly data collection → Feast feature store → "
        "Random Forest / Ridge models → 24h/48h/72h forecasts"
    )
    st.markdown("---")
    st.markdown("**🎨 EPA AQI Scale**")
    scale = [
        ("0–50", "Good", "#00e400"),
        ("51–100", "Moderate", "#dbc400"),
        ("101–150", "Unhealthy (Sensitive)", "#ff7e00"),
        ("151–200", "Unhealthy", "#ff0000"),
        ("201–300", "Very Unhealthy", "#8f3f97"),
        ("301–500", "Hazardous", "#7e0023"),
    ]
    for rng, label, color in scale:
        st.markdown(
            f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:50%;margin-right:8px;flex-shrink:0;"></span>'
            f'<span style="font-size:0.85rem;">{rng} — {label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Built by Waleed | Serverless ML Project")

# ================= LOAD DATA & MODELS =================
df = load_historical_data()
models = load_models()
predictions, latest_row = make_predictions(df, models)

current_aqi = df["aqi"].iloc[-1]
current_time = df["timestamp"].iloc[-1]
cat, color = aqi_category(current_aqi)

# ================= HEADER =================
st.title(f"🌫️ {CITY} Air Quality Forecast")
st.caption(f"Last updated: {current_time.strftime('%d %B %Y, %I:%M %p')}")

# ================= CURRENT AQI =================
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.metric("Current AQI", f"{current_aqi:.0f}")
    st.markdown(
        f'<span class="category-badge" style="background:{color};">{cat}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### Dominant readings right now")
    latest = latest_row.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ Temp", f"{latest['temperature']:.0f}°C")
    m2.metric("💧 Humidity", f"{latest['humidity']:.0f}%")
    m3.metric("🌬️ Wind", f"{latest['wind']:.0f} km/h")
    m4.metric("📊 Pressure", f"{latest['pressure']:.0f} hPa")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= ALERT =================
max_forecast = max(predictions.values())
if max_forecast > 150:
    st.error(f"⚠️ **Hazardous conditions expected within 72 hours** — peak forecast AQI: {max_forecast:.0f}. Limit outdoor activity.")
elif max_forecast > 100:
    st.warning(f"⚠️ Unhealthy-for-sensitive-groups levels expected — peak forecast AQI: {max_forecast:.0f}.")
else:
    st.success(f"✅ Air quality expected to stay in a manageable range over the next 72 hours (peak: {max_forecast:.0f}).")

# ================= FORECAST CARDS =================
st.subheader("📅 3-Day Forecast")
cols = st.columns(3)
for i, horizon in enumerate([24, 48, 72]):
    pred_aqi = predictions[horizon]
    pred_cat, pred_color = aqi_category(pred_aqi)
    with cols[i]:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"**+{horizon} hours**")
        st.metric(label="", value=f"{pred_aqi:.0f}", delta=f"{pred_aqi - current_aqi:+.0f} vs now")
        st.markdown(
            f'<span class="category-badge" style="background:{pred_color};">{pred_cat}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ================= CHART =================
st.subheader("📈 AQI Trend: History & Forecast")
recent = df.tail(24 * 7)[["timestamp", "aqi"]].copy()
recent["type"] = "Historical"

forecast_rows = [
    {"timestamp": current_time + timedelta(hours=h), "aqi": predictions[h], "type": "Forecast"}
    for h in [24, 48, 72]
]
forecast_df = pd.DataFrame(forecast_rows)

chart_df = pd.concat([recent, forecast_df], ignore_index=True)
chart_pivot = chart_df.pivot_table(index="timestamp", columns="type", values="aqi")
st.line_chart(chart_pivot, color=[accent, "#ff7e00"])

# ================= RAW DATA =================
with st.expander("🔍 View recent raw data"):
    st.dataframe(df.tail(20), use_container_width=True)

st.markdown("---")

# ================= SHAP EXPLANATION =================
st.subheader("🧠 Model Explainability (SHAP)")
st.caption("Which features drive each forecast the most")
horizon_choice = st.selectbox("Select forecast horizon:", [24, 48, 72], format_func=lambda h: f"{h} hours ahead")
img_col1, img_col2 = st.columns(2)
with img_col1:
    st.image(f"models/shap_importance_{horizon_choice}h.png", caption="Feature importance (average impact)")
with img_col2:
    st.image(f"models/shap_summary_{horizon_choice}h.png", caption="Feature impact distribution")
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
        return "Good", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "#eab308"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#f97316"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7"
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


def dominant_pollutant(latest):
    """Return the pollutant with the highest normalized reading, for display."""
    pollutant_map = {
        "pm25": "PM2.5", "pm10": "PM10", "o3": "O3",
        "no2": "NO2", "so2": "SO2", "co": "CO",
    }
    # rough normalization thresholds so CO (measured in different units) doesn't always dominate
    norm_ref = {"pm25": 35, "pm10": 150, "o3": 100, "no2": 100, "so2": 40, "co": 9}
    scores = {k: latest[k] / norm_ref[k] for k in pollutant_map}
    top = max(scores, key=scores.get)
    return pollutant_map[top]


def gauge_svg(value, max_value, accent, text_color, subtext_color, track_color):
    """Big, clean AQI number display (no arc)."""
    value = max(0, min(value, max_value))

    html = f"""
    <div style="text-align:center;padding:6px 0;">
        <div style="font-size:3.0rem;font-weight:800;line-height:1;color:{text_color};font-family:'Inter',sans-serif;">{value:.0f}</div>
        <div style="font-size:0.8rem;letter-spacing:1.5px;color:{subtext_color};font-family:'Inter',sans-serif;margin-top:6px;">AQI</div>
    </div>
    """
    return html


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")

# ================= THEME STATE =================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

if is_dark:
    bg_color = "#0b0e14"
    card_bg = "#12161f"
    border_color = "#232838"
    text_color = "#f4f6fb"
    subtext_color = "#8b93a7"
    accent = "#4fc3f7"
    track_color = "#232838"
    chip_bg = "#1a1f2b"
    banner_bg = "#1c1f14"
    banner_border = "#3a3a1a"
else:
    bg_color = "#f6f8fb"
    card_bg = "#ffffff"
    border_color = "#e7eaf0"
    text_color = "#111827"
    subtext_color = "#6b7280"
    accent = "#0284c7"
    track_color = "#eef1f6"
    chip_bg = "#eef2ff"
    banner_bg = "#fff8e6"
    banner_border = "#f4e2ad"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{ background-color: {bg_color}; }}

#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
    padding-top: 2rem;
    max-width: 1150px;
}}

h1, h2, h3, h4, p, span, label, div,
[data-testid="stSidebar"] *,
[data-testid="stMarkdownContainer"] * {{
    color: {text_color};
}}

[data-testid="stCaptionContainer"] * {{
    color: {subtext_color} !important;
}}

[data-testid="stMetricValue"] {{ color: {text_color}; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ color: {subtext_color}; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* card containers */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}

/* pill badge at the very top */
.eyebrow-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {chip_bg};
    color: {accent};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
    text-transform: uppercase;
}}

.page-title {{
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}}

.page-subtitle {{
    color: {subtext_color};
    font-size: 0.95rem;
    margin-bottom: 1.6rem;
    max-width: 640px;
}}

.category-badge {{
    display: inline-block;
    padding: 5px 16px;
    border-radius: 999px;
    font-weight: 600;
    color: white;
    font-size: 0.82rem;
}}

.refresh-box {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 10px 16px;
    text-align: right;
    font-size: 0.8rem;
}}
.refresh-box .label {{
    color: {subtext_color};
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.68rem;
}}
.refresh-box .value {{
    font-weight: 700;
    font-size: 0.95rem;
}}

.alert-banner {{
    background: {banner_bg};
    border: 1px solid {banner_border};
    border-radius: 14px;
    padding: 14px 18px;
    margin: 6px 0 22px 0;
    font-size: 0.9rem;
}}

.legend-row {{
    display:flex; align-items:center; margin-bottom:6px;
}}
.legend-dot {{
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:8px; flex-shrink:0;
}}
.legend-text {{ font-size:0.83rem; color:{subtext_color}; }}

/* sidebar */
[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
    border-right: 1px solid {border_color};
}}
[data-testid="stSidebar"] > div {{
    background-color: {card_bg} !important;
}}
[data-testid="stSidebarUserContent"] {{
    background-color: {card_bg} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: {border_color};
}}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("### 🌫️ AQI Forecast")
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
        ("0–50", "Good", "#22c55e"),
        ("51–100", "Moderate", "#eab308"),
        ("101–150", "Unhealthy (Sensitive)", "#f97316"),
        ("151–200", "Unhealthy", "#ef4444"),
        ("201–300", "Very Unhealthy", "#a855f7"),
        ("301–500", "Hazardous", "#7e0023"),
    ]
    for rng, label, color in scale:
        st.markdown(
            f'<div class="legend-row">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'<span class="legend-text">{rng} — {label}</span></div>',
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
latest = latest_row.iloc[0]
dom_pollutant = dominant_pollutant(latest)

# ================= HEADER =================
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown('<div class="eyebrow-chip">🌫️ Live AQI Intelligence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{CITY} Air Quality Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">3-day AQI forecast blending gradient boosting with a persistence '
        'baseline, retrained daily · data refreshed hourly</div>',
        unsafe_allow_html=True,
    )
with head_col2:
    st.markdown(
        f'''<div class="refresh-box">
                <div class="label">Last refresh</div>
                <div class="value">{current_time.strftime('%d %b %Y')}</div>
                <div class="label">{current_time.strftime('%I:%M %p')}</div>
            </div>''',
        unsafe_allow_html=True,
    )

# ================= CURRENT AQI =================
col1, col2 = st.columns([1, 2])
with col1:
    with st.container(border=True):
        gcol1, gcol2 = st.columns([1, 1])
        with gcol1:
            st.markdown(
                gauge_svg(current_aqi, 300, color, text_color, subtext_color, track_color),
                unsafe_allow_html=True,
            )
        with gcol2:
            st.markdown(
                f'<div style="padding-top:14px;">'
                f'<span class="category-badge" style="background:{color};">{cat}</span>'
                f'<div style="margin-top:10px;font-size:0.8rem;color:{subtext_color};">'
                f'Dominant pollutant<br><b style="color:{text_color};font-size:0.95rem;">{dom_pollutant}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
with col2:
    with st.container(border=True):
        st.markdown("#### Dominant readings right now")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ Temp", f"{latest['temperature']:.0f}°C")
        m2.metric("💧 Humidity", f"{latest['humidity']:.0f}%")
        m3.metric("🌬️ Wind", f"{latest['wind']:.0f} km/h")
        m4.metric("📊 Pressure", f"{latest['pressure']:.0f} hPa")

# ================= ALERT =================
max_forecast = max(predictions.values())
alert_icon = "🔴" if max_forecast > 150 else ("🟠" if max_forecast > 100 else "🟢")
if max_forecast > 150:
    msg = f"Hazardous conditions expected within 72 hours — peak forecast AQI: <b>{max_forecast:.0f}</b>. Limit outdoor activity."
elif max_forecast > 100:
    msg = f"Unhealthy-for-sensitive-groups levels expected — peak forecast AQI: <b>{max_forecast:.0f}</b>. Sensitive groups (children, elderly, respiratory/heart conditions) should limit prolonged outdoor exertion."
else:
    msg = f"Air quality expected to stay in a manageable range over the next 72 hours (peak: <b>{max_forecast:.0f}</b>)."

st.markdown(f'<div class="alert-banner">{alert_icon} {msg}</div>', unsafe_allow_html=True)

# ================= FORECAST CARDS =================
st.subheader("📅 3-Day Forecast")
cols = st.columns(3)
for i, horizon in enumerate([24, 48, 72]):
    pred_aqi = predictions[horizon]
    pred_cat, pred_color = aqi_category(pred_aqi)
    with cols[i]:
        with st.container(border=True):
            st.markdown(f"**+{horizon} hours**")
            st.metric(label="", value=f"{pred_aqi:.0f}", delta=f"{pred_aqi - current_aqi:+.0f} vs now")
            st.markdown(
                f'<span class="category-badge" style="background:{pred_color};">{pred_cat}</span>',
                unsafe_allow_html=True,
            )

# ================= CHART =================
st.subheader("📈 AQI Trend: History & Forecast")
with st.container(border=True):
    recent = df.tail(24 * 7)[["timestamp", "aqi"]].copy()
    recent["type"] = "Historical"

    forecast_rows = [
        {"timestamp": current_time + timedelta(hours=h), "aqi": predictions[h], "type": "Forecast"}
        for h in [24, 48, 72]
    ]
    forecast_df = pd.DataFrame(forecast_rows)

    chart_df = pd.concat([recent, forecast_df], ignore_index=True)
    chart_pivot = chart_df.pivot_table(index="timestamp", columns="type", values="aqi")
    st.line_chart(chart_pivot, color=[accent, "#f97316"])

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
    with st.container(border=True):
        st.image(f"models/shap_importance_{horizon_choice}h.png", caption="Feature importance (average impact)")
with img_col2:
    with st.container(border=True):
        st.image(f"models/shap_summary_{horizon_choice}h.png", caption="Feature impact distribution")
# 🌫️ Lahore AQI Forecasting Pipeline

A serverless machine learning pipeline that predicts Lahore's Air Quality Index (AQI) up to 72 hours in advance.

**🔗 Live Dashboard:** [aqi-forecast-tbkhapelqeb2kgiere7kpn.streamlit.app](https://aqi-forecast-tbkhapelqeb2kgiere7kpn.streamlit.app/)

## Tech Stack

Python · scikit-learn · SHAP · Feast · Streamlit · GitHub Actions · IQAir API

## Running Locally

```bash
git clone https://github.com/waleedusman747/aqi-forecast.git
cd aqi-forecast
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:


Then:
```bash
cd aqi_repo/feature_repo && feast apply && cd ../..
python feature_pipeline.py
streamlit run app.py
```

## Author

Waleed — BSIT, Bahria University Lahore
import requests
import os
from dotenv import load_dotenv

print("Script start...")

load_dotenv()
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

print("Token :", AQICN_TOKEN)

CITY = "lahore"
url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"

print("Request sent...")
response = requests.get(url)

print("Status code:", response.status_code)
print("Raw response:", response.text)
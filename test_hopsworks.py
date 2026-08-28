import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

print("Connecting to Hopsworks...")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)

print("Connected successfully!")
print("Project name:", project.name)
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-94ae4b7e-fec2.cloud.databricks.com/serving-endpoints"
)

def predict_fn(text:str) -> str:

    response = client.chat.completions.create(
        model="gemini_3_1_flash_Newer",
        messages=[
            {
                "role": "user",
                "content":text
            }
        ],
        max_tokens=5000
    )
    return response.choices[0].message.content
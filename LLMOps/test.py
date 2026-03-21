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
  base_url=os.environ.get('GEMINI_ENDPOINT')
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "Tell me about Large Language Models"
  }
  ],
  model="gemini_3_1_flash_Newer",
  max_tokens=256
)

print(chat_completion.choices[0].message.content)
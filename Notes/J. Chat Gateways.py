from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import os

# To get a DATABRICKS_TOKEN, click the "Generate Access Token" button or follow https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://1314887821841093.ai-gateway.cloud.databricks.com/mlflow/v1"
)

chat_completion = client.chat.completions.create(
  messages=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is Databricks?"},
  ],
  model="gemini_aigateway",
  max_tokens=1024
)

# print(chat_completion.choices[0].message.content)
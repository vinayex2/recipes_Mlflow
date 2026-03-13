from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# To get a DATABRICKS_TOKEN, click the "Generate Access Token" button or follow https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN environment variable not set or empty. Please set your Databricks token before running.")

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://1314887821841093.ai-gateway.cloud.databricks.com/openai/v1"
)

response = client.responses.create(
  model="gemini_codex",
  max_output_tokens=256,
  input=[
    {
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello!"}]
    },
    {
      "role": "assistant",
      "content": [{"type": "output_text", "text": "Hello! How can I assist you today?"}]
    },
    {
      "role": "user",
      "content": [{"type": "input_text", "text": "What is Databricks?"}]
    }
  ]
)

print(response.output_text)
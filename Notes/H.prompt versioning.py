# %pip install --upgrade "mlflow[databricks]>=3.1.0"
# %pip install openai
# %restart_python

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Register a prompt template
prompt_name = "mlflow_experiments.genai.assistant_prompt"
prompt = mlflow.genai.register_prompt(
    name=prompt_name,
    template="You are a helpful assistant. Answer this question: {{question}}",
    commit_message="Initial customer support prompt"
)
print(f"Created version {prompt.version}")  # "Created version 1"

# Set a production alias
mlflow.genai.set_prompt_alias(
    name=prompt_name,
    alias="production",
    version=1
)

# Load and use the prompt in your application
prompt = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{prompt_name}@production")
# response = llm.invoke(prompt.format(question="How do I reset my password?"))
print(prompt.format(question="How do I reset my password?"))

client = OpenAI()

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages = [
        {'role' :'user',
         'content': prompt.format(question="How do I reset my password?")
         }
    ]
)

print(response)
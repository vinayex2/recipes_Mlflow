%pip install --upgrade "mlflow[databricks]>=3.1.0"
%restart_python

import mlflow

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
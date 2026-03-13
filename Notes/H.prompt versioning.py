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

# client = OpenAI()

# response = client.chat.completions.create(
#     model='gpt-4o-mini',
#     messages = [
#         {'role' :'user',
#          'content': prompt.format(question="How do I reset my password?")
#          }
#     ]
# )

# print(response)

def predict_fn(question):
    """Define how to invoke the model"""
    output = None
    # try:    
    #     response = client.chat.completions.create(
    #         model='gpt-4o-mini',
    #         messages = [
    #             {'role' :'user',
    #             'content': prompt.format(question=question)
    #             }
    #         ])
    #     output = response.choices[0].message.content
    # except:
    #     print("Error")
    output = "Default Output"
    
    return output

#Evaluation of Model Output
data = [
    {
        'inputs': {"question": "who invented telephone"},
        'expectations': {"expected_response": "alexander graham bell"}
    },
    {
        'inputs': {"question": "who was the inventor of radium"},
        'expectations': {"expected_response": "marie curie"}
    },
    {
        'inputs': {"question": "who founded apple"},
        'expectations': {"expected_response": "Default Output"}
    }

]

from mlflow.genai import scorer
from mlflow.entities import Feedback

@scorer
def exact_match_scorer(inputs, outputs, expectations):
    """Define how to evaluate the model output"""
    score = 0
    if outputs == expectations['expected_response']:
        score = 1
    return score


mlflow.genai.evaluate(
    data=data,
    predict_fn=predict_fn,
    scorers = [
        mlflow.genai.scorers.Correctness(),
        exact_match_scorer,
        mlflow.genai.scorers.Guidelines(name='is_professional', guidelines="Responses should be professional and respectful."),
        # mlflow.genai.scorers.Relevance(name='is_relevant', relevance_threshold=0.7)
    ]
)


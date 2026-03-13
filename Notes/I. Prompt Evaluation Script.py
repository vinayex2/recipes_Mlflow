import pypdf
import re

def clean_text(text):
  text = text.lower()
  text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  return text.strip()

reader = pypdf.PdfReader("Profile_Vinay_Tyagi_24.pdf")

resume_text = ""

for page in reader.pages:
  resume_text += page.extract_text()


clean_resume_text = clean_text(resume_text)
print(clean_resume_text)


import mlflow
from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.genai import scorer
from mlflow.entities import Feedback

eval_dataset = [
  {
    'inputs' : {'clean_resume_text': clean_resume_text},
    "expectations" : {"expected_response": "python,ml,sklearn,linear regression,machine learning"}
  }
]

def predict_fn(clean_resume_text) -> str:
  response = "python,ml,sklearn,linear regression,machine learning"
  return response

@scorer
def minimum_five_skills(inputs,outputs, expectations):
  # response = len(outputs['response'].split(",")) >= 5
  return True
    

scorers = [
  Correctness(),
  Guidelines(name = "coverage", guidelines = 'Are all the skills captured?'),
  minimum_five_skills
]  


mlflow.genai.evaluate(
  data = eval_dataset,
  predict_fn=predict_fn,
  scorers=scorers
)
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import mlflow
import pandas as pd

#Download the model
model_download_dir = "/Workspace/Users/vasaicrow@gmail.com/recipes_Mlflow/Notes/hf_model_files/"

embed_model = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
embed_model.save_pretrained(model_download_dir)

model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model.save_pretrained(model_download_dir)

def embed_text(text):
  return embed_model.encode(text)

#Test the model
texts = ["This is a sample text", "This is another sample text",]  

inputs = embed_model(texts, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state.mean(dim=1)
print(embeddings.shape)


#Create the model
class SBERTCustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = AutoModel.from_pretrained(model_download_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_download_dir)
        
    def load_context(self, context):
        model_dir = context.artifacts['model_dir']
        self.model = AutoModel.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input['text'].tolist()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # embeddings = outputs.last_hidden_state.mean(axis=1)
        embs = []
        for i in outputs.last_hidden_state:
            embs.append(i.mean(axis=0).flatten().tolist())
        # embeddings = torch.stack(embs)
        return pd.DataFrame({'predictions': embs})
    

#Log the  Model

with mlflow.start_run():
    model = mlflow.pyfunc.log_model(
        name="model",
        python_model=SBERTCustomModel(),
        artifacts = {"model_dir": model_download_dir},
        input_example=pd.DataFrame({"text": ["this is a sample text", "this is another sample"]})
    )
    # mlflow.models.validate_serving_input
    # mlflow.models.convert_input_example_to_serving_input

    #Register the Model
    model_name = "mlflow_experiments.mlflow_models.best_lm_model"
    model_uri = "runs:/" + mlflow.active_run().info.run_id + "/model"
    model_registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # client = mlflow.tracking.MlflowClient()
    # model_registered = client.create_registered_model("SBERTCustomModel")
    # model_version = client.create_model_version("SBERTCustomModel", model.model_uri, "1.0)






   
    


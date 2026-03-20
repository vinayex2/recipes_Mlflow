import os
import requests
import numpy as np
import pandas as pd
import json
from dotenv import load_dotenv
load_dotenv()

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://dbc-94ae4b7e-fec2.cloud.databricks.com/serving-endpoints/sbert_model/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

print(score_model(pd.DataFrame({'text': ['This is a test sentence', 'This is another sentence']})))


#Spark UDF for data operations
from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf('string')
def score_model_udf(dataset):
    return pd.Series(score_model(dataset)['predictions'])    

spark.udf.register("score_model", score_model_udf)



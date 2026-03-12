import mlflow

model_uri = "models:/mlflow_experiments.mlflow_models.best_lr_model/2"
mlflowmodel = mlflow.sklearn.load_model(model_uri)
print(mlflowmodel)
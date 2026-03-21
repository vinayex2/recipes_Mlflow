import mlflow

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.89)

# 1314887821841093
print(mlflow.get_tracking_uri())
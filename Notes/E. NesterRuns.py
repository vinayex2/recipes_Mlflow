import mlflow

#Can be used for hyperparameter runs
with mlflow.start_run(run_name="parent run") as parent_run:
    print(f"Parent Run: {parent_run.info.run_id}")
    mlflow.log_param('theta',"100")
    with mlflow.start_run(run_name='child run 1' , nested=True) as childrun:
        print(f"Child Run: {childrun.info.run_id}")
        mlflow.log_param('theta',"100")

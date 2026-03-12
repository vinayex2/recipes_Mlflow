import mlflow

# Model Lineage, Versioning and Aliasing

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

with mlflow.start_run(run_name="sklearn model logging"):
    # Load the Iris dataset
    # Enable autologging for scikit-learn
    # mlflow.sklearn.autolog()
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 500,
        "random_state": 7777,
    }
    # mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    mlflow.sklearn.log_model(lr, "LR_model", input_example=X_train[[0]], 
                             registered_model_name='mlflow_experiments.mlflow_models.best_lr_model')

    # Make predictions on the test set

    
    print("Training Complete")
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random
import numpy as np
import dagshub
dagshub.init(repo_owner='d4tu4llam', repo_name='Workflow-CI', mlflow=True)
#tracking UI
mlflow.set_tracking_uri("https://dagshub.com/d4tu4llam/Workflow-CI.mlflow")
mlflow.set_experiment("Submission SML Modelling Diabetes Prediction Hilmi Datu Allam")

train_data = pd.read_csv("diabetes_prediction_dataset_preprocessing/train_processed.csv")
test_data = pd.read_csv("diabetes_prediction_dataset_preprocessing/test_processed.csv")

# Extract features and target from train data
X_train = train_data.drop("diabetes", axis=1)
y_train = train_data["diabetes"]

# Extract features from test data
X_test = test_data.drop("diabetes", axis=1)
y_test = test_data["diabetes"]

input_example = X_train[0:5]



with mlflow.start_run():
    #Log parameters
    n_estimators = 5
    max_depth =5
    mlflow.autolog()
    #Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model = model,
        artifact_path = "model",
        input_example = input_example
    )
    
    #Evaluate model
    predictions = model.predict(X_test)
    accuracy= model.score(X_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

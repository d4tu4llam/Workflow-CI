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
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Submission SML Modelling Tuning Diabetes Prediction Hilmi Datu Allam")
train_data = pd.read_csv("diabetes_prediction_dataset_preprocessing/train_processed.csv")
test_data = pd.read_csv("diabetes_prediction_dataset_preprocessing/test_processed.csv")

# Extract features and target from train data
X_train = train_data.drop("diabetes", axis=1)
y_train = train_data["diabetes"]

# Extract features from test data
X_test = test_data.drop("diabetes", axis=1)
y_test = test_data["diabetes"]

#Ukuran input
input_example = X_train[0:5]

#Parameter untuk utning
n_estimators_range = np.linspace(10,300, 5, dtype=int) #akan naik incremental 5
max_depth_range = np.linspace(1,50, 5, dtype=int)#akan naik incremental 5
best_accuracy = 0
best_params={}



for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name =f"elastic_search_{n_estimators}_{max_depth}"):
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth if max_depth else "None")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("tuning_method", "Manual Grid Search")
            #Train model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state =42)
            model.fit(X_train, y_train)

            #Evaluate model
            predictions = model.predict(X_test)
            accuracy= model.score(X_test, y_test)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            
            # Log metrics
            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            #Save the best model
            if accuracy> best_accuracy:
                best_accuracy =accuracy
                best_params={n_estimators:n_estimators, max_depth:max_depth}
                mlflow.sklearn.log_model(
                    sk_model = model,
                    artifact_path ="model",
                    input_example=input_example

                )

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from mlflow.client import MlflowClient
import random
import numpy as np
from mlflow.models import infer_signature


def log_all_metrics(y_train_pred, y_train_proba, y_test_pred ,y_test_proba ):
    # Log all metrics
    mlflow.log_metrics({
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "train_auc": roc_auc_score(y_train, y_train_proba),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "test_auc": roc_auc_score(y_test, y_test_proba),
        "test_log_loss": log_loss(y_test, y_test_proba),
    })
 
def log_metric_info():
    #Buat metric_info.json
    metric_info = {
        "metrics": [
            {"name": "test_accuracy",        "type": "scalar"},
            {"name": "test_precision",       "type": "scalar"},
            {"name": "test_recall",          "type": "scalar"},
            {"name": "test_f1",              "type": "scalar"},
            {"name": "test_auc",             "type": "scalar"},
            {"name": "test_log_loss",        "type": "scalar"},
            {"name": "training_accuracy",    "type": "scalar"},
            {"name": "training_precision",   "type": "scalar"},
            {"name": "training_recall",      "type": "scalar"},
            {"name": "training_f1",          "type": "scalar"},
            {"name": "training_auc",         "type": "scalar"},
            {"name": "training_log_loss",    "type": "scalar"},
            {"name": "RandomForestClassifier_score_X_test", "type": "scalar"}
        ]
    }

    # Simpan ke file JSON
    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=2)

    # Log ke MLflow
    mlflow.log_artifact("metric_info.json")
    os.remove("metric_info.json")  

def log_estimator_html(model):
    #Buat estimator.html
    html_content = model._repr_html_()   

    # Simpan ke file
    with open("estimator.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # Log ke MLflow
    mlflow.log_artifact("estimator.html")
    os.remove("estimator.html")

def log_visual(model, X_test, y_test_pred, y_test_proba):
    # estimator.html
    with open("estimator.html", "w", encoding="utf-8") as f:
        f.write(model._repr_html_())
    mlflow.log_artifact("estimator.html")
    os.remove("estimator.html")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f"Confusion Matrix (")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix.png"); plt.close()
    mlflow.log_artifact(f"confusion_matrix.png")
    os.remove(f"confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_val:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"roc_curve.png"); plt.close()
    mlflow.log_artifact(f"roc_curve.png")
    os.remove(f"roc_curve.png")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"precision_recall_curve.png"); plt.close()
    mlflow.log_artifact(f"precision_recall_curve.png")
    os.remove(f"precision_recall_curve.png")

if __name__ == "__main__":
    # BISA PAKAI ARGUMEN ATAU DEFAULT (PERSIS KAYAK MENTOR!)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    # Path dataset otomatis
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "diabetes_prediction_dataset_preprocessing")
    
    train_path = os.path.join(data_folder, "train_processed.csv")
    test_path = os.path.join(data_folder, "test_processed.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Extract features and target from train data
    X_train = train_data.drop("diabetes", axis=1)
    y_train = train_data["diabetes"]

    # Extract features from test data
    X_test = test_data.drop("diabetes", axis=1)
    y_test = test_data["diabetes"]

    input_example = X_train.iloc[0:5].values


    with mlflow.start_run():
        #Log parameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
            "bootstrap": True,
            "class_weight": None,
            "criterion": "gini"
        }
        mlflow.log_params(params)

        #Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        #Training Metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        #Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        #Log all metrics
        log_all_metrics(y_train_pred, y_train_proba, y_test_pred, y_test_proba)
        #Log metric info
        log_metric_info()
        #Log Estimator
        log_estimator_html(model)
        #Log visual seperti cm roc auc 
        log_visual(model, X_test, y_test_pred, y_test_proba)

        
        signature = infer_signature(X_test, y_test_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="Diabetes_RFC_Hilmi_Manual"
        )

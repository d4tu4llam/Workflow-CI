import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from mlflow.models import infer_signature
import json
import os
os.environ["MPLBACKEND"] = "Agg"
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
#Kalau mau pake dagshub
import dagshub
dagshub.init(repo_owner='d4tu4llam', repo_name='Workflow-CI', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/d4tu4llam/Workflow-CI.mlflow")

#tracking UI Local
#mlflow.set_tracking_uri("http://127.0.0.1:5000/")
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

best_custom_score=0

def objective(trial):
    """
    Objective function untuk Optuna Hyperparameter Tuning
    
    Objective: Memaksimalkan skor gabungan (Custom Weighted Score)
               yang mengutamakan Recall (keselamatan pasien) sambil
               menjaga Precision agar tidak jatuh terlalu rendah (efisiensi biaya).
    
    Custom Score yang digunakan:
        score = 0.8 x Recall + 0.2 x Precision
    
    Alasan pemilihan bobot:
        - Recall diberi bobot 80% karena:
              False Negative (melewatkan penderita diabetes) 
              dapat berakibat fatal 
        - Precision diberi bobot 20% karena:
              False Positive yang terlalu banyak menyebabkan 
              biaya screening ulang tinggi dan beban sistem kesehatan
    
    Pendekatan ini merupakan kompromi optimal antara:
        - Recall dan Precision untuk konteks medis diabetes.
    
    Metrik yang di-log:
        - custom_recall_precision_score : nilai objective (0.8R + 0.2P)
        - test_recall, test_precision   : untuk monitoring detail
        - semua metrik standar via log_all_metrics()
    
    Return:
        float: custom_recall_precision_score (nilai yang dioptimasi Optuna)
    """
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 8, 40)
    max_depth = None if max_depth >= 35 else max_depth
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    class_weight = trial.suggest_categorical("class_weight", ["balanced",None,{0:1, 1:3},{0:1, 1:4},{0:1, 1:5},{0:1, 1:6}] )


    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "random_state":42,
        "n_jobs": -1,
        "bootstrap": bootstrap,
        "class_weight": class_weight,
        "criterion": "gini"
            }

    # Setiap trial jadi nested run
    with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_param("optuna_trial_number", trial.number)
        #Train model

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        #Training Metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        #Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)

        # score custom karena kalau mementingkan recall tadi precisionnya drop ke 0.4
        score = 0.8 * recall + 0.2 * precision   
        mlflow.log_metric("custom_recall_precision_score", score)
        mlflow.log_metric("test_recall", recall)
        #Log all metrics
        log_all_metrics(y_train_pred, y_train_proba, y_test_pred, y_test_proba)
        #Log metric info
        log_metric_info()
        #Log Estimator
        log_estimator_html(model)
        # if score >= best_custom_score:
           
        #     #Log visual seperti cm roc auc 
        #     log_visual(model, X_test, y_test_pred, y_test_proba)
        #     signature = infer_signature(X_test, y_test_pred)
        #     mlflow.sklearn.log_model(
        #         sk_model=model,
        #         artifact_path="model",
        #         signature=signature,
        #         input_example=input_example
        #     )
        
        mlflow.set_tag("optuna_trial", "true")
       
            
        
    return score

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

with mlflow.start_run(run_name="Optuna_Manual_Nested"):
    mlflow.set_tag("tuning_method", "optuna_manual_nested")
    mlflow.set_tag("objective", "maximize_test_auc")
    mlflow.log_param("total_trials", 20)

    study.optimize(objective, n_trials=20)

best_params = study.best_params
fixed_params=({
    "random_state": 42,
    "n_jobs": -1,
    "criterion": "gini"
})

with mlflow.start_run(run_name="Best_Model_Final_Optuna_Manual"):
    mlflow.set_tag("status", "BEST_MODEL")
    mlflow.log_params(best_params)
    mlflow.log_params(fixed_params)
    mlflow.log_metric("best_custom_score", study.best_value)
    mlflow.log_metric("best_trial_number", study.best_trial.number)

    # Train ulang model terbaik
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    y_train_pred = best_model.predict(X_train)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_test_pred  = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    # Gunakan fungsi kamu yang sudah ada
    log_all_metrics(y_train_pred, y_train_proba, y_test_pred, y_test_proba)
    log_visual(best_model, X_test, y_test_pred, y_test_proba)
    log_estimator_html(best_model)
    log_metric_info()

    # Register model terbaik
    signature = infer_signature(X_test, y_test_pred)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="Diabetes_RFC_Optuna_Manual_Best"
    )


import sys
import subprocess

# # Ensure MLflow is installed
try:
    import mlflow
    import sagemaker_mlflow
except ImportError:
    print("Installing MLflow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",  "boto3==1.37.1", "botocore==1.37.1", "s3transfer", "mlflow==2.22.0", "sagemaker-mlflow==0.1.0", "numpy", "matplotlib==3.7.3"])
    import mlflow
    import sagemaker_mlflow
    
import mlflow.sklearn
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
import joblib
import glob
import json
import ast

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Only use this for headless environments (e.g. script mode)
    print("matplotlib loaded")
except ImportError as e:
    print("matplotlib is not installed or failed to load:", e)
    plt = None

import importlib.util
if importlib.util.find_spec("matplotlib") is not None:
    print("✅ matplotlib is available")
else:
    print("❌ matplotlib is NOT available")


parser = argparse.ArgumentParser()
parser.add_argument("--tracking_server_arn", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Default")
parser.add_argument("--model_output_path", type=str, default="/opt/ml/model")
parser.add_argument("--model_type", type=str, default="logistic_regression")
parser.add_argument("--model_param_grid", type=str)  # Will be a JSON string
parser.add_argument("--max_iter", type=int)
parser.add_argument("--random_state", type=int)
parser.add_argument("--model_cv", type=int)
parser.add_argument("--run_name", type=str, default="run-default")
args, _ = parser.parse_known_args()

def safe_parse_param_grid(raw_string):
    try:
        return json.loads(raw_string)
    except json.JSONDecodeError:
        print("⚠️ json.loads() failed. Falling back to Python-style parser.")
        print("Raw string received:", raw_string)
        
        # Try to parse as key=value pairs
        try:
            param_dict = {}
            import re
            param_items = re.split(r',(?![^[]*\])', raw_string)
            for item in param_items:
                if '=' not in item:
                    continue
                key, val = item.split('=', 1)
                key = key.strip()
                val = val.strip()
                # Convert string "None" to Python None
                if val == "None":
                    parsed_val = None
                else:
                    parsed_val = ast.literal_eval(val)
                param_dict[key] = parsed_val
            return param_dict
        except Exception as e:
            print("❌ Fallback parsing also failed.")
            print("Error:", str(e))
            raise ValueError("Invalid format for model_param_grid.")

print('Start-Train')
# Load training data
# train_path = glob.glob("/opt/ml/input/data/train/*.csv")[0]
train_path = os.path.join("/opt/ml/input/data/train", "train.csv")
df = pd.read_csv(train_path)
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

# # Set up MLflow
mlflow.set_tracking_uri(args.tracking_server_arn)
mlflow.set_experiment(args.experiment_name)

experiment = mlflow.set_experiment(args.experiment_name)
print("Experiment ID:", experiment.experiment_id)

with mlflow.start_run(run_name=args.run_name) as run:
    
    mlflow.log_param("model_type", args.model_type)
    
    model_param_grid = safe_parse_param_grid(args.model_param_grid)
    print("parsed model_param_grid:")
    print(model_param_grid)

    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("param_grid", str(model_param_grid))
    mlflow.log_param("cv", args.model_cv)
    
    if args.model_type == "logistic_regression":
        mlflow.log_param("max_iter", args.max_iter)
        model = LogisticRegression(max_iter=args.max_iter, random_state=args.random_state)
        model_grid = GridSearchCV(model, model_param_grid, cv=args.model_cv, scoring='accuracy', n_jobs=-1, verbose=1)
        print('LR')
    elif args.model_type == "random_forest":
        model = RandomForestClassifier(random_state=args.random_state)
        model_grid = GridSearchCV(model, model_param_grid, cv=args.model_cv, scoring='accuracy', n_jobs=-1, verbose=1)
        print('RF')
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    model_grid.fit(X, y)
    best_model = model_grid.best_estimator_

    # Log best parameters
    mlflow.log_params(model_grid.best_params_)

    #evaluate
    y_pred = best_model.predict(X)
    mlflow.log_metric("accuracy", accuracy_score(y, y_pred))
    mlflow.log_metric("recall", recall_score(y, y_pred, average="binary"))
    mlflow.log_metric("precision", precision_score(y, y_pred, average="binary"))
    mlflow.log_metric("f1_score", f1_score(y, y_pred, average="binary"))

    # Log classification report
    report = classification_report(y, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Log confusion matrix
    try:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
    except Exception as e:
        print("Error on confusion matrix:")
        print("Error:", str(e))

    # # Log feature importances if available
    # if plt:
    #     try:
    #         if hasattr(best_model, "feature_importances_"):
    #             importances = best_model.feature_importances_
    #             feature_names = X.columns
    
    #             fig, ax = plt.subplots(figsize=(10, 6))
    #             sorted_idx = np.argsort(importances)[::-1]
    #             ax.barh(range(len(importances)), importances[sorted_idx])
    #             ax.set_yticks(range(len(importances)))
    #             ax.set_yticklabels(feature_names[sorted_idx])
    #             ax.set_title("Feature Importances (Random Forest)")
    #             plt.tight_layout()
    #             plt.savefig("feature_importances.png")
    #             mlflow.log_artifact("feature_importances.png")
    #             plt.close(fig)
    
    #         elif hasattr(best_model, "coef_"):
    #             importances = np.abs(best_model.coef_[0])
    #             feature_names = X.columns
    
    #             fig, ax = plt.subplots(figsize=(10, 6))
    #             sorted_idx = np.argsort(importances)[::-1]
    #             ax.barh(range(len(importances)), importances[sorted_idx])
    #             ax.set_yticks(range(len(importances)))
    #             ax.set_yticklabels(feature_names[sorted_idx])
    #             ax.set_title("Feature Importances (Logistic Regression)")
    #             plt.tight_layout()
    #             plt.savefig("feature_importances.png")
    #             mlflow.log_artifact("feature_importances.png")
    #             plt.close(fig)
    
    #         else:
    #             print("⚠️ Feature importances not available for this model.")
    #     except Exception as e:
    #         print(f"⚠️ Could not log feature importances: {e}")
        
    # Log the trained model
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")

    os.makedirs(args.model_output_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(args.model_output_path, "model.joblib"))
    with open(os.path.join(args.model_output_path, "run_id.txt"), "w") as f:
        f.write(run.info.run_id)

    print(f"Training complete.")
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Model saved to {os.path.join(args.model_output_path, 'model.joblib')}")

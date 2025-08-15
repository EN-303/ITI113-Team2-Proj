
import sys
import subprocess

# # Ensure MLflow is installed
try:
    import mlflow
    import sagemaker_mlflow
except ImportError:
    print("Installing MLflow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",  "boto3==1.37.1", "botocore==1.37.1", "s3transfer", "mlflow==2.22.0", "sagemaker-mlflow==0.1.0"])
    import mlflow
    import sagemaker_mlflow
    
import mlflow.sklearn
import os
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--tracking_server_arn", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Default")
parser.add_argument("--model_output_path", type=str, default="/opt/ml/model")
parser.add_argument("-C", "--C", type=float, default=0.5)
parser.add_argument("--run_name", type=str, default="Experiment-LR")
args, _ = parser.parse_known_args()

print('Start-Train')
# Load training data
train_path = glob.glob("/opt/ml/input/data/train/*.csv")[0]
df = pd.read_csv(train_path)
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

# # Set up MLflow
mlflow.set_tracking_uri(args.tracking_server_arn)
mlflow.set_experiment(args.experiment_name)

with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_param("C", args.C)
    model = LogisticRegression(C=args.C)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    os.makedirs(args.model_output_path, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_output_path, "model.joblib"))
    with open(os.path.join(args.model_output_path, "run_id.txt"), "w") as f:
        f.write(run.info.run_id)

    print(f"Training complete. Accuracy: {acc:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")

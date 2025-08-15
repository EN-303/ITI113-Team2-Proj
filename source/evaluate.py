import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os
import json
import boto3
import tarfile

if __name__ == "__main__":
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the directory containing the model.tar.gz file.")
    parser.add_argument("--test-path", type=str, required=True, help="Path to the directory containing test.csv.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the evaluation.json report.")
    parser.add_argument("--model-package-group-name", type=str, required=True, help="Name of the SageMaker Model Package Group.")
    parser.add_argument("--region", type=str, required=True, help="The AWS region for creating the boto3 client.")
    args = parser.parse_args()

    # --- Extract and Load Model ---
    # SageMaker packages models in a .tar.gz file. We need to extract it first.
    model_archive_path = os.path.join(args.model_path, 'model.tar.gz')
    print(f"Extracting model from archive: {model_archive_path}")
    with tarfile.open(model_archive_path, "r:gz") as tar:
        tar.extractall(path=args.model_path)

    # Load the model using joblib
    model_file_path = os.path.join(args.model_path, "model.joblib")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file 'model.joblib' not found after extraction in: {args.model_path}")
    
    print(f"Loading model from: {model_file_path}")
    model = joblib.load(model_file_path)

    # --- Prepare Data and Evaluate ---
    test_file_path = os.path.join(args.test_path, "test.csv")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test data not found: {test_file_path}")
    
    test_df = pd.read_csv(test_file_path)
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    
    print("Running predictions on the test dataset.")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = {"accuracy": accuracy}
    print(f"Calculated accuracy: {accuracy:.4f}")

    # --- Check for Existing Baseline Model in SageMaker Model Registry ---
    print(f"Checking for baseline model in region: {args.region}")
    sagemaker_client = boto3.client("sagemaker", region_name=args.region)
    try:
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=args.model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        # If the list is not empty, an approved model already exists
        report["baseline_exists"] = len(response["ModelPackageSummaryList"]) > 0
        if report["baseline_exists"]:
            print(f"An approved baseline model was found in '{args.model_package_group_name}'.")
        else:
             print(f"No approved baseline model was found in '{args.model_package_group_name}'.")

    except sagemaker_client.exceptions.ClientError as e:
        # If the ModelPackageGroup doesn't exist, there is no baseline
        if "ResourceNotFound" in str(e):
            report["baseline_exists"] = False
            print(f"Model Package Group '{args.model_package_group_name}' not found. Assuming no baseline exists.")
        else:
            raise

    # --- Write Final Report ---
    os.makedirs(args.output_path, exist_ok=True)
    report_path = os.path.join(args.output_path, "evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Evaluation complete. Report written to: {report_path}")
    print("Evaluation Report:")
    print(json.dumps(report, indent=4))

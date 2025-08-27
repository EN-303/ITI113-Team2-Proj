import subprocess
import sys
import os


# --- Install required packages ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3==1.28.57", "botocore==1.31.57", "numpy==1.24.1", "sagemaker" ])

# Ensure sagemaker SDK is installed before importing
try:
    import sagemaker
except ImportError:
    print("sagemaker SDK not found. Installing now...")
    install("sagemaker")
    import sagemaker

import argparse
import sagemaker
import boto3
from sagemaker.model import Model
import shutil
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept the registered model's ARN instead of the S3 data path
    parser.add_argument("--model-package-arn", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sm_client = boto3.client("sagemaker", region_name=args.region)

    # --- Step 1: Get Model Artifacts from the Model Package ---
    print(f"Describing model package: {args.model_package_arn}")
    model_package_description = sm_client.describe_model_package(ModelPackageName=args.model_package_arn)
    
    # Extract the S3 path to the model artifacts (model.tar.gz)
    model_artifacts = model_package_description["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    # Extract the container image URI
    image_uri = model_package_description["InferenceSpecification"]["Containers"][0]["Image"]

    print(f"Found model artifacts at: {model_artifacts}")
    print(f"Using container image: {image_uri}")

    # --- Step 2: Prepare a clean directory for the inference code ---
    original_code_location = "/opt/ml/processing/input/scripts"
    inference_script_path = os.path.join(original_code_location, "inference.py")
    clean_code_dir = "/tmp/code"

    if not os.path.exists(inference_script_path):
        raise FileNotFoundError(f"inference.py not found at {inference_script_path}. Did you include it via ProcessingInput?")
    
    # Create the clean directory, removing it first if it exists
    if os.path.exists(clean_code_dir):
        shutil.rmtree(clean_code_dir)
    os.makedirs(clean_code_dir)

    # Copy only the inference script to the clean directory
    shutil.copy(inference_script_path, clean_code_dir)
    print(f"Copied inference.py to clean dir: {clean_code_dir}")
    
    # --- Step 2: Create a SageMaker Model object using the local inference.py ---
    # This explicitly tells SageMaker to use your provided inference script.
    model = Model(
        image_uri=image_uri,
        model_data=model_artifacts, # Use artifacts from the registered model
        role=args.role,
        sagemaker_session=sagemaker_session,
        entry_point="inference.py",  # Explicitly use your inference script
        source_dir=clean_code_dir         # Directory containing inference.py
    )
   
    # First, try to delete existing resources to ensure a clean deployment
    try:
        # Delete the endpoint first
        sm_client.delete_endpoint(EndpointName=args.endpoint_name)
        print(f"Deleted existing endpoint: {args.endpoint_name}")
        
        # Then, delete the endpoint config
        sm_client.delete_endpoint_config(EndpointConfigName=args.endpoint_name)
        print(f"Deleted existing endpoint config: {args.endpoint_name}")
    except sm_client.exceptions.ClientError as e:
        # If the resources don't exist, that's fine.
        if "Could not find" not in str(e):
            raise e
    
    # Deploy the model to an endpoint
    bucket_name = 'iti113-team2-bucket'
    base_folder = 'Team2'
    print(f"Deploying registered model from ARN to endpoint: {args.endpoint_name}")

    try:
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,
            destination_s3_uri=f"s3://{bucket_name}/{base_folder}/monitoring/data-capture",
            # csv_content_types=['text/csv'],
            json_content_types=['application/json'],
            sagemaker_session=sagemaker_session
        )

        model.deploy(
            initial_instance_count=1,
            instance_type="ml.t2.medium",
            endpoint_name=args.endpoint_name,
            data_capture_config=data_capture_config,
            # Update endpoint if it already exists
            update_endpoint=True,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        print("Deployment complete.")
        
    except Exception as e:
        print(f"Error model.deploy: {e}")
        
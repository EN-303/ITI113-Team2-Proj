import os
import joblib
import json
import pandas as pd

def model_fn(model_dir):

    print("Loading model from a .joblib file.")
    # The model is saved as 'model.joblib' in your training script.
    model_path = os.path.join(model_dir, "model.joblib")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def input_fn(request_body, request_content_type):

    print(f"Received request of type: {request_content_type}")
    if request_content_type == 'application/json':
        try:
            # Assuming the JSON input is in the format: {"data": [[...], [...]]}
            data = json.loads(request_body)
            
            if "data" not in data or not isinstance(data["data"], list):
                raise ValueError("JSON must contain a 'data' field with a list of row dictionaries.")

            df = pd.DataFrame(data['data'])
            return df
            
        except Exception as e:
            raise ValueError(f"Error parsing JSON: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):

    print("Making predictions on the input data.")
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

def output_fn(prediction, response_content_type):
    print(f"Serializing prediction to: {response_content_type}")
    if response_content_type == 'application/json':
        try:
            # Convert numpy array to a list and wrap it in a JSON object.
            response = {"predictions": prediction.tolist()}
            return json.dumps(response)
        except Exception as e:
            raise ValueError(f"Error serializing prediction to JSON: {e}")
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

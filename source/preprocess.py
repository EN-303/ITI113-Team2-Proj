
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
import joblib
import shutil

def build_preprocessor(numeric_cols, categorical_cols):
    print('preprocess-transformer-start')
    
    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Although no missing values, it's good practice
    ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create a preprocessor to apply the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    print('preprocess-transformer-end')

    return preprocessor
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Directory containing Team2Dataset.csv")
    parser.add_argument("--output-train-path", type=str, help="Output directory for train.csv")
    parser.add_argument("--output-test-path", type=str, help="Output directory for test.csv")
    parser.add_argument("--output-transformer-path", type=str, help="Output directory for preprocessor.pkl")
    args = parser.parse_args()

    # Use provided paths or fall back to SageMaker defaults
    input_path = args.input_path or "/opt/ml/processing/input"
    output_train_path = args.output_train_path or "/opt/ml/processing/train"
    output_test_path = args.output_test_path or "/opt/ml/processing/test"
    output_transformer_path = args.output_transformer_path or "/opt/ml/processing/artifacts"

    input_file = os.path.join(input_path, "Team2Dataset.csv")
    print(f"Reading input file from {input_file}...")
    df = pd.read_csv(input_file)
    # df = preprocess(df) #clean data
    
    df.drop(columns=["patientid"], inplace=True) #drop column
    df.drop(columns=['age'], inplace=True) #drop column
    
    # Define categorical and numerical features for preprocessing
    categorical_features = ['gender', 'chestpain', 'fastingbloodsugar', 'restingrelectro', 'exerciseangia', 'slope', 'noofmajorvessels']
    numerical_features = ['restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']

    X = df.drop(columns=["target"])
    y = df["target"]

    # Fit transformer
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)

    # Save the transformer
    os.makedirs(output_transformer_path, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_transformer_path, "preprocessor.pkl"))

    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_features)

    # Combine back into DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    X_transformed_df["target"] = y.values
    
    print("Splitting into train/test...")
    train, test = train_test_split(X_transformed_df, test_size=0.2, random_state=42)

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    train_output = os.path.join(output_train_path, "train.csv")
    test_output = os.path.join(output_test_path, "test.csv")

    print(f"Saving train to {train_output}")
    train.to_csv(train_output, index=False)

    print(f"Saving test to {test_output}")
    test.to_csv(test_output, index=False)

    # # Archive original input file to transformer (artifact) output
    # input_archive_dir = os.path.join(output_transformer_path, "input_archive")
    # os.makedirs(input_archive_dir, exist_ok=True)
    
    # try:
    #     archived_file_path = os.path.join(input_archive_dir, os.path.basename(input_file))
    #     shutil.move(input_file, archived_file_path)
    #     print(f"Archived input file to: {archived_file_path}")
    # except Exception as e:
    #     print(f"Warning: Failed to archive input file: {e}")
    
    
    print("Preprocessing completed!")
    print("End")

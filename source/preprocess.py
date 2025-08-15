
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess(df):
    print('preprocess-start')
    print(f"Dataset shape: {df.shape}")

    #drop patient identifier
    df.drop(columns=['patientid'], inplace=True)
    
    df['oldpeak'] = df['oldpeak'].apply(lambda x: 0 if x < 0 else x)
    df['oldpeak_log'] = np.log1p(df['oldpeak'])
    df.drop(columns=['oldpeak'], inplace=True)
    print('transform oldpeak-end')
    
    num_cols = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak_log']
    for col in num_cols:
        #Replace invalid zeros with NaN 
        if col in ['restingBP', 'serumcholestrol']:  # zero Cholesterol/RestingBP is invalid
            df[col] = df[col].replace(0, np.nan)
        
        #Impute NaNs with median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
        #IQR-based Outlier Capping (Winsorization)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap the outliers to within the IQR bounds
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print('impute and remove outlier-end')

    #To group categorical variable
    # df['ChestPainType_Grouped'] = df['ChestPainType'].replace({'TA': 'Other', 'ATA': 'Other'})
    # df = df.drop(columns=['ChestPainType'])
    # df = pd.get_dummies(df, columns=['ChestPainType_Grouped'], drop_first=False)
    
    #One-Hot Encoding on categorical variable
    df = pd.get_dummies(df, columns=['chestpain'], drop_first=False) #Keep all dummy columns
    df = pd.get_dummies(df, columns=['restingrelectro'], drop_first=False)
    df = pd.get_dummies(df, columns=['slope'], drop_first=False)
    df = pd.get_dummies(df, columns=['noofmajorvessels'], drop_first=False) 
    print('encoding-end')
    
    print(f"Dataset: {df.head(2)}")
    print('preprocess-end')

    return df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="Directory containing Team2Dataset.csv")
    parser.add_argument("--output-train-path", type=str, help="Output directory for train.csv")
    parser.add_argument("--output-test-path", type=str, help="Output directory for test.csv")
    args = parser.parse_args()

    # Use provided paths or fall back to SageMaker defaults
    input_path = args.input_path or "/opt/ml/processing/input"
    output_train_path = args.output_train_path or "/opt/ml/processing/train"
    output_test_path = args.output_test_path or "/opt/ml/processing/test"

    input_file = os.path.join(input_path, "Team2Dataset.csv")
    print(f"Reading input file from {input_file}...")
    df = pd.read_csv(input_file)
    df = preprocess(df) #clean data
    
    print("Splitting into train/test...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    train_output = os.path.join(output_train_path, "train.csv")
    test_output = os.path.join(output_test_path, "test.csv")

    print(f"Saving train to {train_output}")
    train.to_csv(train_output, index=False)

    print(f"Saving test to {test_output}")
    test.to_csv(test_output, index=False)

    print("Preprocessing complete.")

"""
data_preprocessing.py

This script performs data preprocessing steps including handling missing values, encoding categorical variables,
normalizing numerical features, and preparing the data for further analysis and modeling.

Author: Satej
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def handle_missing_values(data):
    """
    Handles missing values in the dataset by imputing or dropping them.

    Args:
        data (pd.DataFrame): Input data with potential missing values.

    Returns:
        pd.DataFrame: Data after handling missing values.
    """
    # Example: Fill missing values in numerical columns with the mean
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Drop rows where essential categorical columns are missing
    data = data.dropna(subset=['department', 'job_role'])
    print("Missing values handled.")
    return data

def encode_categorical_variables(data):
    """
    Encodes categorical variables using label encoding.

    Args:
        data (pd.DataFrame): Input data containing categorical variables.

    Returns:
        pd.DataFrame: Data with encoded categorical variables.
    """
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        label_encoders[column] = encoder
        print(f"Encoded column: {column}")

    return data, label_encoders

def normalize_features(data, columns_to_normalize):
    """
    Normalizes specified numerical features using Min-Max scaling.

    Args:
        data (pd.DataFrame): Input data containing numerical features.
        columns_to_normalize (list): List of column names to be normalized.

    Returns:
        pd.DataFrame: Data with normalized features.
    """
    scaler = MinMaxScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    print("Numerical features normalized.")
    return data, scaler

if __name__ == "__main__":
    # Example usage
    INPUT_FILE = "satej_extracted_employee_data.csv"  # Path to the extracted data
    OUTPUT_FILE = "satej_preprocessed_employee_data.csv"  # Path to save the preprocessed data

    # Load the data
    data = pd.read_csv(INPUT_FILE)
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Handle missing values
    data = handle_missing_values(data)

    # Encode categorical variables
    data, encoders = encode_categorical_variables(data)

    # Normalize numerical features
    columns_to_normalize = ['age', 'overtime_hours', 'years_at_company']
    data, scaler = normalize_features(data, columns_to_normalize)

    # Save the preprocessed data
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Preprocessed data saved to {OUTPUT_FILE}.")

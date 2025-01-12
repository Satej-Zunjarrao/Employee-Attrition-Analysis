"""
feature_engineering.py

This script performs feature engineering to create new features, scale numerical data,
and prepare the dataset for predictive modeling.

Author: Satej
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_tenure_group(data):
    """
    Creates a new feature categorizing employees based on their tenure.

    Args:
        data (pd.DataFrame): Input data containing the 'years_at_company' column.

    Returns:
        pd.DataFrame: Data with the new 'tenure_group' column.
    """
    bins = [0, 2, 5, 10, 20]
    labels = ['0-2 years', '2-5 years', '5-10 years', '10+ years']
    data['tenure_group'] = pd.cut(data['years_at_company'], bins=bins, labels=labels)
    print("Tenure group feature added.")
    return data

def calculate_overtime_percentage(data):
    """
    Creates a new feature calculating overtime as a percentage of total work hours.

    Args:
        data (pd.DataFrame): Input data containing 'overtime_hours' and 'years_at_company' columns.

    Returns:
        pd.DataFrame: Data with the new 'overtime_percentage' column.
    """
    # Assume average total work hours per year is 2080 (40 hours/week * 52 weeks)
    total_hours_per_year = 2080
    data['overtime_percentage'] = (data['overtime_hours'] / total_hours_per_year) * 100
    print("Overtime percentage feature calculated.")
    return data

def scale_numerical_features(data, columns_to_scale):
    """
    Scales numerical features using Min-Max scaling.

    Args:
        data (pd.DataFrame): Input data containing numerical features.
        columns_to_scale (list): List of column names to scale.

    Returns:
        pd.DataFrame: Data with scaled features.
    """
    scaler = MinMaxScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    print("Numerical features scaled.")
    return data, scaler

if __name__ == "__main__":
    # Example usage
    INPUT_FILE = "satej_preprocessed_employee_data.csv"  # Path to preprocessed data
    OUTPUT_FILE = "satej_engineered_employee_data.csv"  # Path to save the engineered data

    # Load the data
    data = pd.read_csv(INPUT_FILE)
    print(f"Data loaded for feature engineering with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Add tenure group feature
    data = add_tenure_group(data)

    # Calculate overtime percentage
    data = calculate_overtime_percentage(data)

    # Scale numerical features
    columns_to_scale = ['age', 'overtime_hours', 'overtime_percentage', 'years_at_company']
    data, scaler = scale_numerical_features(data, columns_to_scale)

    # Save the engineered data
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Feature-engineered data saved to {OUTPUT_FILE}.")

"""
dashboard_visualization.py

This script prepares data visualizations for a Tableau dashboard by exporting key insights and summaries.
It focuses on creating datasets for heatmaps, key predictors, and individual attrition risk.

Author: Satej
"""

import pandas as pd

def create_heatmap_data(data, group_by_columns, target_column):
    """
    Prepares data for heatmaps by calculating attrition rates for specific groups.

    Args:
        data (pd.DataFrame): Input dataset.
        group_by_columns (list): Columns to group data by (e.g., ['department', 'job_role']).
        target_column (str): Target column indicating attrition (binary: 0/1).

    Returns:
        pd.DataFrame: Heatmap data with attrition rates.
    """
    heatmap_data = data.groupby(group_by_columns)[target_column].mean().reset_index()
    heatmap_data.rename(columns={target_column: "attrition_rate"}, inplace=True)
    print("Heatmap data prepared.")
    return heatmap_data

def create_predictor_importance_data(model, feature_columns):
    """
    Extracts feature importance from a trained Random Forest model.

    Args:
        model: Trained Random Forest model.
        feature_columns (list): List of feature column names.

    Returns:
        pd.DataFrame: Feature importance data sorted by importance.
    """
    importances = model.feature_importances_
    importance_data = pd.DataFrame({"feature": feature_columns, "importance": importances})
    importance_data.sort_values(by="importance", ascending=False, inplace=True)
    print("Predictor importance data prepared.")
    return importance_data

def save_data_for_tableau(data, file_name):
    """
    Saves data to a CSV file for Tableau integration.

    Args:
        data (pd.DataFrame): Data to be saved.
        file_name (str): File name for the CSV file.

    Returns:
        None
    """
    data.to_csv(file_name, index=False)
    print(f"Data saved for Tableau: {file_name}")

if __name__ == "__main__":
    INPUT_FILE = "satej_engineered_employee_data.csv"  # Path to engineered data
    RF_MODEL_FILE = "satej_random_forest_model.pkl"  # Trained Random Forest model file
    HEATMAP_OUTPUT = "satej_attrition_heatmap.csv"  # Heatmap data for Tableau
    PREDICTOR_IMPORTANCE_OUTPUT = "satej_predictor_importance.csv"  # Feature importance data for Tableau

    # Load the data
    data = pd.read_csv(INPUT_FILE)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Load the trained Random Forest model
    import pickle
    with open(RF_MODEL_FILE, "rb") as file:
        rf_model = pickle.load(file)

    # Prepare heatmap data
    heatmap_data = create_heatmap_data(data, group_by_columns=["department", "job_role"], target_column="attrition")
    save_data_for_tableau(heatmap_data, HEATMAP_OUTPUT)

    # Prepare predictor importance data
    predictor_importance_data = create_predictor_importance_data(rf_model, feature_columns=data.columns[:-1])
    save_data_for_tableau(predictor_importance_data, PREDICTOR_IMPORTANCE_OUTPUT)

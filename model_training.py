"""
model_training.py

This script trains classification models (Random Forest and Logistic Regression) to predict employee attrition.
It saves the trained models for later evaluation and deployment.

Author: Satej
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle  # For saving models to disk

def split_data(data, target_column):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): Name of the target column.

    Returns:
        tuple: X_train, X_test, y_train, y_test (training and testing splits).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained.")
    return rf_model

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train, y_train)
    print("Logistic Regression model trained.")
    return lr_model

if __name__ == "__main__":
    INPUT_FILE = "satej_engineered_employee_data.csv"  # Path to engineered data
    TARGET_COLUMN = "attrition"  # Target column
    RF_MODEL_FILE = "satej_random_forest_model.pkl"  # File to save the Random Forest model
    LR_MODEL_FILE = "satej_logistic_regression_model.pkl"  # File to save the Logistic Regression model

    # Load the data
    data = pd.read_csv(INPUT_FILE)
    print(f"Data loaded for training with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, TARGET_COLUMN)

    # Train models
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)

    # Save models to disk
    with open(RF_MODEL_FILE, "wb") as rf_file:
        pickle.dump(rf_model, rf_file)
    print(f"Random Forest model saved to {RF_MODEL_FILE}.")

    with open(LR_MODEL_FILE, "wb") as lr_file:
        pickle.dump(lr_model, lr_file)
    print(f"Logistic Regression model saved to {LR_MODEL_FILE}.")

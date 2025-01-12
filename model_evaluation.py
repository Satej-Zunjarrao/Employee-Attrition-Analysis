"""
model_evaluation.py

This script evaluates trained classification models on a test dataset using metrics such as accuracy,
precision, recall, and F1-score. It also generates a confusion matrix for visual assessment.

Author: Satej
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(model_file):
    """
    Loads a trained model from disk.

    Args:
        model_file (str): Path to the saved model file.

    Returns:
        sklearn model: Loaded model.
    """
    with open(model_file, "rb") as file:
        model = pickle.load(file)
    print(f"Model loaded from {model_file}.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints performance metrics.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, F1-score).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred):
    """
    Plots a confusion matrix.

    Args:
        y_test (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.

    Returns:
        None
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Attrition", "Attrition"], yticklabels=["No Attrition", "Attrition"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    INPUT_FILE = "satej_engineered_employee_data.csv"  # Path to engineered data
    RF_MODEL_FILE = "satej_random_forest_model.pkl"  # Random Forest model file
    LR_MODEL_FILE = "satej_logistic_regression_model.pkl"  # Logistic Regression model file
    TARGET_COLUMN = "attrition"  # Target column

    # Load the data
    data = pd.read_csv(INPUT_FILE)
    print(f"Data loaded for evaluation with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Split the data
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Evaluate Random Forest model
    rf_model = load_model(RF_MODEL_FILE)
    rf_metrics, rf_predictions = evaluate_model(rf_model, X_test, y_test)
    plot_confusion_matrix(y_test, rf_predictions)

    # Evaluate Logistic Regression model
    lr_model = load_model(LR_MODEL_FILE)
    lr_metrics, lr_predictions = evaluate_model(lr_model, X_test, y_test)
    plot_confusion_matrix(y_test, lr_predictions)

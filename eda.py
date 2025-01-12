"""
eda.py

This script performs Exploratory Data Analysis (EDA) to uncover trends, correlations, and key insights
related to employee attrition. It generates visualizations and statistical summaries.

Author: Satej
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition_rate(data):
    """
    Plots the overall attrition rate in the dataset.

    Args:
        data (pd.DataFrame): Input data containing the 'attrition' column.

    Returns:
        None
    """
    attrition_counts = data['attrition'].value_counts(normalize=True)
    attrition_counts.plot(kind='bar', color=['blue', 'orange'])
    plt.title("Attrition Rate")
    plt.xlabel("Attrition")
    plt.ylabel("Proportion")
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
    plt.show()

def plot_categorical_trends(data, column):
    """
    Plots attrition rates for a specific categorical column.

    Args:
        data (pd.DataFrame): Input data.
        column (str): Column name to analyze.

    Returns:
        None
    """
    grouped_data = data.groupby(column)['attrition'].mean().sort_values()
    grouped_data.plot(kind='bar', color='green')
    plt.title(f"Attrition Rate by {column.capitalize()}")
    plt.xlabel(column.capitalize())
    plt.ylabel("Attrition Rate")
    plt.xticks(rotation=45)
    plt.show()

def correlation_heatmap(data):
    """
    Generates a heatmap of correlations between numerical features.

    Args:
        data (pd.DataFrame): Input data containing numerical features.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    # Example usage
    INPUT_FILE = "satej_preprocessed_employee_data.csv"  # Path to preprocessed data

    # Load the data
    data = pd.read_csv(INPUT_FILE)
    print(f"Data loaded for EDA with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Plot overall attrition rate
    plot_attrition_rate(data)

    # Plot attrition rates by department
    plot_categorical_trends(data, "department")

    # Plot attrition rates by job role
    plot_categorical_trends(data, "job_role")

    # Generate a heatmap of numerical feature correlations
    correlation_heatmap(data)

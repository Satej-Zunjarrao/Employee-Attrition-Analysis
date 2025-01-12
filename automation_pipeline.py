"""
automation_pipeline.py

This script automates the entire data pipeline, from extraction to dashboard visualization preparation.
It ensures that updated data is processed, analyzed, and prepared for use in predictive modeling and Tableau.

Author: Satej
"""

import os
import subprocess

def run_script(script_name):
    """
    Executes a Python script in the current environment.

    Args:
        script_name (str): Name of the Python script to execute.

    Returns:
        None
    """
    try:
        print(f"Running script: {script_name}")
        subprocess.run(["python", script_name], check=True)
        print(f"Script {script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_name}: {e}")

if __name__ == "__main__":
    # Define the sequence of scripts to run
    scripts = [
        "data_extraction.py",       # Extract data from the database
        "data_preprocessing.py",    # Preprocess the extracted data
        "feature_engineering.py",   # Perform feature engineering
        "model_training.py",        # Train models
        "dashboard_visualization.py"  # Prepare data for Tableau dashboard
    ]

    # Execute each script sequentially
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Script {script} not found. Skipping...")

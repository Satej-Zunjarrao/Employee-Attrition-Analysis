"""
data_extraction.py

This script connects to the HR database, executes SQL queries to extract relevant employee data,
and saves the data into a CSV file for further processing.

Author: Satej
"""

import sqlite3  # Importing sqlite3 for database connection (can replace with another database library if needed)
import pandas as pd

def extract_data(database_path, query, output_file):
    """
    Connects to the database, executes the SQL query, and saves the data to a CSV file.

    Args:
        database_path (str): Path to the HR database file.
        query (str): SQL query to extract required employee data.
        output_file (str): Path to save the extracted data as a CSV file.

    Returns:
        None
    """
    try:
        # Connect to the database
        connection = sqlite3.connect(database_path)
        print("Database connection successful.")

        # Execute the query and load data into a pandas DataFrame
        data = pd.read_sql_query(query, connection)
        print(f"Query executed successfully. Retrieved {len(data)} rows.")

        # Save the data to a CSV file
        data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Ensure the database connection is closed
        connection.close()
        print("Database connection closed.")

if __name__ == "__main__":
    # Example usage
    DATABASE_PATH = "/path/to/hr_database.db"  # Replace with your actual database path
    SQL_QUERY = """
    SELECT
        employee_id,
        department,
        job_role,
        age,
        gender,
        performance_rating,
        overtime_hours,
        promotion_last_5_years,
        years_at_company,
        attrition
    FROM
        employee_data;
    """
    OUTPUT_FILE = "satej_extracted_employee_data.csv"  # Output CSV file

    extract_data(DATABASE_PATH, SQL_QUERY, OUTPUT_FILE)

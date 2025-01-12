# Employee Attrition Analysis System

## Overview
The **Employee Attrition Analysis System** is a Python-based solution designed to identify factors contributing to employee attrition and predict employees at risk of leaving the organization. By leveraging predictive modeling, data visualization, and automation, this system enables HR teams to implement proactive strategies to improve retention and workforce management.

This project includes a modular and scalable pipeline for:
- Data extraction
- Cleaning
- Exploratory analysis
- Feature engineering
- Predictive modeling
- Visualization
- Automation

---

## Key Features
- **Data Extraction**: Retrieves employee data from HR databases using SQL.
- **Data Cleaning**: Handles missing values, encodes categorical variables, and normalizes numerical features.
- **Exploratory Data Analysis (EDA)**: Visualizes trends in attrition across demographics and job roles.
- **Feature Engineering**: Creates additional features like tenure group and overtime percentage.
- **Predictive Modeling**: Trains Random Forest and Logistic Regression models for attrition prediction.
- **Visualization**: Prepares heatmaps and feature importance datasets for Tableau dashboards.
- **Automation**: Automates the entire pipeline to ensure real-time updates.

## Directory Structure
```
project/
│
├── data_extraction.py          # Extracts employee data from HR databases
├── data_preprocessing.py       # Cleans and preprocesses the data
├── eda.py                      # Generates visualizations and insights
├── feature_engineering.py      # Creates additional features for modeling
├── model_training.py           # Trains predictive models for attrition analysis
├── model_evaluation.py         # Evaluates model performance and metrics
├── dashboard_visualization.py  # Prepares data for Tableau dashboards
├── automation_pipeline.py      # Automates the end-to-end pipeline
├── README.md                   # Project documentation
```

## Modules

1. **data_extraction.py**
   - Extracts employee data from SQL databases.
   - Saves extracted data in CSV format for preprocessing.

2. **data_preprocessing.py**
   - Handles missing values and encodes categorical variables.
   - Normalizes numerical features to improve model performance.

3. **eda.py**
   - Visualizes attrition rates by department and job role.
   - Generates heatmaps for correlations among features.

4. **feature_engineering.py**
   - Creates additional features like tenure group and overtime percentage.
   - Prepares data for predictive modeling.

5. **model_training.py**
   - Trains classification models (Random Forest and Logistic Regression).
   - Saves trained models for evaluation and deployment.

6. **model_evaluation.py**
   - Evaluates model performance using accuracy, precision, recall, and F1-score.
   - Generates confusion matrices for visual performance assessment.

7. **dashboard_visualization.py**
   - Prepares datasets for Tableau dashboards.
   - Creates heatmaps of attrition rates and feature importance datasets.

8. **automation_pipeline.py**
   - Automates the pipeline from data extraction to visualization.
   - Executes scripts sequentially to ensure seamless updates.

---

## Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com  

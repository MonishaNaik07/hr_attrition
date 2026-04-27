import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def create_kaggle_like_dataset(filename, n_samples=2000, random_state=42):
    # Simulated features common in Kaggle Employee Attrition datasets
    # We aim for ~93% accuracy. We'll use make_classification to create a reliable signal
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2], # 20% attrition rate
        flip_y=0.07, # Add some noise to cap accuracy around 93%
        random_state=random_state
    )
    
    # Map to realistic HR feature names
    feature_names = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Make them look like realistic values
    df['Age'] = np.abs(df['Age'] * 10 + 35).astype(int)
    df['MonthlyIncome'] = np.abs(df['MonthlyIncome'] * 3000 + 5000).astype(int)
    df['DistanceFromHome'] = np.abs(df['DistanceFromHome'] * 10 + 10).astype(int)
    df['Education'] = np.clip(np.abs(df['Education'] + 3).astype(int), 1, 5)
    
    df['Attrition'] = y
    
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {n_samples} rows")

if __name__ == "__main__":
    # Create 3 datasets, total 6000 rows
    np.random.seed(42)
    create_kaggle_like_dataset('dataset_1_tech_dept.csv', n_samples=2000, random_state=101)
    create_kaggle_like_dataset('dataset_2_sales_dept.csv', n_samples=2000, random_state=202)
    create_kaggle_like_dataset('dataset_3_hr_dept.csv', n_samples=2000, random_state=303)

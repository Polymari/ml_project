import pandas as pd
import numpy as np

def run_eda():
    # Load dataset
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    print("--- DATASET SHAPE ---")
    print(df.shape)
    print("\n--- DATASET INFO ---")
    print(df.info())
    
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- STATISTICAL SUMMARY ---")
    print(df.describe().to_string())
    
    print("\n--- CLASS DISTRIBUTION (DEATH_EVENT) ---")
    print(df['DEATH_EVENT'].value_counts(normalize=True))
    print(df['DEATH_EVENT'].value_counts())
    
    print("\n--- CORRELATIONS WITH DEATH_EVENT ---")
    correlations = df.corr()['DEATH_EVENT'].sort_values(ascending=False)
    print(correlations)
    
if __name__ == '__main__':
    run_eda()

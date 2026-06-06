import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
import joblib

def train_and_save():
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    # 1. Feature Engineering
    df['creatinine_sodium_ratio'] = df['serum_creatinine'] / df['serum_sodium']
    df['is_elderly'] = (df['age'] >= 65).astype(int)
    
    # 2. Train XGBoost Classifier (excluding time)
    X = df.drop(columns=['DEATH_EVENT', 'time'])
    y = df['DEATH_EVENT']
    
    ratio = (len(y) - sum(y)) / sum(y)
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            random_state=42, 
            eval_metric='logloss', 
            scale_pos_weight=ratio,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=100
        ))
    ])
    
    xgb_pipeline.fit(X, y)
    joblib.dump(xgb_pipeline, 'heart_failure_xgb_pipeline.joblib')
    print("XGBoost pipeline saved successfully.")
    
    # 3. Train Cox Proportional Hazards Model (including time and DEATH_EVENT)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='time', event_col='DEATH_EVENT')
    joblib.dump(cph, 'heart_failure_cph_model.joblib')
    print("CoxPH model saved successfully.")

if __name__ == '__main__':
    train_and_save()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline

def run_baselines():
    # Load dataset
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    # Target and features
    y = df['DEATH_EVENT']
    
    # Case 1: Including 'time' (Leakage case)
    X_with_time = df.drop(columns=['DEATH_EVENT'])
    
    # Case 2: Excluding 'time' (Realistic clinical case)
    X_no_time = df.drop(columns=['DEATH_EVENT', 'time'])
    
    print("==================================================")
    print("CASE 1: USING ALL FEATURES (INCLUDING 'TIME' - LEAKAGE)")
    print("==================================================")
    evaluate_dataset(X_with_time, y)
    
    print("\n==================================================")
    print("CASE 2: CLINICAL PREDICTION (EXCLUDING 'TIME')")
    print("==================================================")
    evaluate_dataset(X_no_time, y)

def evaluate_dataset(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Models
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        if probs is not None:
            print(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

if __name__ == '__main__':
    run_baselines()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
import joblib

from src.features import add_features


def run_training_pipeline(data_path: str, xgb_out_path: str, cph_out_path: str):
    """Trains and saves the classification and survival models."""
    df = pd.read_csv(data_path)
    df_engineered = add_features(df)

    X = df_engineered.drop(columns=['DEATH_EVENT', 'time'])
    y = df_engineered['DEATH_EVENT']

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
    joblib.dump(xgb_pipeline, xgb_out_path)
    print(f"XGBoost pipeline saved to {xgb_out_path}")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_engineered, duration_col='time', event_col='DEATH_EVENT')
    joblib.dump(cph, cph_out_path)
    print(f"CoxPH model saved to {cph_out_path}")

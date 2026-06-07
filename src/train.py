import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
import joblib

from src.features import add_features

CLUSTER_FEATURES = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction',
    'platelets', 'serum_creatinine', 'serum_sodium'
]


def run_training_pipeline(data_path: str, xgb_out_path: str, cph_out_path: str, kmeans_out_path: str = 'heart_failure_kmeans.joblib'):
    """Trains and saves the classification, survival, and clustering models."""
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

    cluster_scaler = StandardScaler()
    X_continuous = df_engineered[CLUSTER_FEATURES]
    X_continuous_scaled = cluster_scaler.fit_transform(X_continuous)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_continuous_scaled)

    df_clustered = df_engineered.copy()
    df_clustered['Cluster'] = cluster_labels
    profile_cols = CLUSTER_FEATURES + ['DEATH_EVENT']
    cluster_profiles = df_clustered.groupby('Cluster')[profile_cols].mean()

    kmeans_artifact = {
        'model': kmeans,
        'scaler': cluster_scaler,
        'features': CLUSTER_FEATURES,
        'profiles': cluster_profiles,
    }
    joblib.dump(kmeans_artifact, kmeans_out_path)
    print(f"KMeans clustering artifact saved to {kmeans_out_path}")


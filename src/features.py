import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering to the raw heart failure dataset."""
    df = df.copy()
    df['creatinine_sodium_ratio'] = df['serum_creatinine'] / df['serum_sodium']
    df['is_elderly'] = (df['age'] >= 65).astype(int)
    return df

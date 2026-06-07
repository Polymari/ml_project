import shutil
import os
import re


def sync():
    """Syncs deploy-ready files to the deploy directory.

    Copies model artifacts and requirements as-is. For app.py, replaces the
    'from src.features import add_features' import with an inlined version
    of the function so the Space doesn't need the src/ package.
    """
    os.makedirs('deploy', exist_ok=True)

    binary_files = [
        'requirements.txt',
        'heart_failure_xgb_pipeline.joblib',
        'heart_failure_cph_model.joblib',
        'heart_failure_kmeans.joblib'
    ]
    for file in binary_files:
        if os.path.exists(file):
            shutil.copy(file, os.path.join('deploy', file))
            print(f"Synced {file} to deploy/")
        else:
            print(f"Warning: {file} not found. Skipping.")

    inlined_features = (
        "import pandas as pd\n"
        "\n"
        "\n"
        "def add_features(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df = df.copy()\n"
        "    df['creatinine_sodium_ratio'] = df['serum_creatinine'] / df['serum_sodium']\n"
        "    df['is_elderly'] = (df['age'] >= 65).astype(int)\n"
        "    return df\n"
    )

    with open('app.py', 'r') as f:
        app_source = f.read()

    app_source = app_source.replace("from src.features import add_features", inlined_features)

    with open(os.path.join('deploy', 'app.py'), 'w') as f:
        f.write(app_source)
    print("Synced app.py to deploy/ (with inlined features)")


if __name__ == '__main__':
    sync()

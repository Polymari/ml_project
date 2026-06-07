import shutil
import os


def sync():
    """Syncs app.py, requirements.txt, and model joblib files to the deploy directory."""
    os.makedirs('deploy', exist_ok=True)
    files_to_copy = [
        'app.py',
        'requirements.txt',
        'heart_failure_xgb_pipeline.joblib',
        'heart_failure_cph_model.joblib'
    ]
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, os.path.join('deploy', file))
            print(f"Synced {file} to deploy/")
        else:
            print(f"Warning: {file} not found. Skipping.")


if __name__ == '__main__':
    sync()

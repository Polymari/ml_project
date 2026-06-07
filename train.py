from src.train import run_training_pipeline


if __name__ == '__main__':
    run_training_pipeline(
        data_path='heart_failure_clinical_records_dataset.csv',
        xgb_out_path='heart_failure_xgb_pipeline.joblib',
        cph_out_path='heart_failure_cph_model.joblib'
    )
    print("Models trained and exported successfully.")

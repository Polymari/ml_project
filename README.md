# Heart Failure Mortality & Survival Predictor

A machine learning project that predicts heart failure mortality risk and models patient survival probability using clinical data from 299 patients.

## Project Overview

This project implements a dual ML pipeline approach:

- **XGBoost Risk Classifier** — Binary classification of mortality risk using clinical features with GridSearchCV hyperparameter tuning
- **Cox Proportional Hazards Model** — Survival analysis modeling individual patient survival curves over time
- **SHAP Interpretability** — TreeExplainer-based feature attribution for transparent risk explanations
- **Unsupervised Clustering** — K-Means patient stratification with PCA visualization

## Directory Structure

```
ml_project/
├── app.py                       # Gradio web application
├── train.py                     # Model training entrypoint
├── requirements.txt             # Python dependencies
├── heart_failure_clinical_records_dataset.csv
├── heart_failure_xgb_pipeline.joblib
├── heart_failure_cph_model.joblib
├── notebooks/
│   └── heart_failure_mortality_predictor.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py              # Shared feature engineering
│   ├── train.py                 # Training pipeline module
│   └── plots.py                 # Visualization utilities
├── scripts/
│   ├── sync_deploy.py           # Deployment staging utility
│   ├── exploration/             # EDA and baseline scripts
│   └── notebook_builder/        # Notebook generation scripts
└── deploy/                      # Staged files for HF Spaces
```

## Getting Started

### Environment Setup

```bash
uv venv
uv pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

### Run Gradio App Locally

```bash
python app.py
```

### Sync Deployment Files

```bash
python scripts/sync_deploy.py
```

## Dataset

[Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) — 299 patients, 13 clinical features including age, ejection fraction, serum creatinine, serum sodium, and follow-up period.

## Deployment

The Gradio app is deployed as a Hugging Face Space at [fergieee/heart-classifier](https://huggingface.co/spaces/fergieee/heart-classifier).

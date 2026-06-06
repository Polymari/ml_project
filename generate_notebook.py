import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # -------------------------------------------------------------
    # 0. Title & Setup
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Clinical Mortality Predictor for Heart Failure Patients\n",
            "This notebook builds an end-to-end Machine Learning pipeline to predict patient mortality (`DEATH_EVENT`) using clinical records. It is designed to be executed directly in **Google Colab**.\n",
            "\n",
            "### Pipeline Stages:\n",
            "1. **Data Collection & Ingestion**\n",
            "2. **Data Preprocessing & Cleaning**\n",
            "3. **Feature Engineering**\n",
            "4. **Data Splitting**\n",
            "5. **Model Training & Tuning**\n",
            "6. **Model Evaluation**\n",
            "7. **Deployment & Monitoring**"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required libraries (for local or Google Colab environments)\n",
            "!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib"
        ]
    })
    
    # -------------------------------------------------------------
    # 1. Data Collection & Ingestion
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Collection & Ingestion\n",
            "Gathering raw data from various sources (databases, APIs, sensors).\n",
            "\n",
            "In this section, we ingest the dataset directly from a public raw GitHub URL. This ensures reproducibility and ease of running in Google Colab."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Raw CSV file path from dimikara/Survival-Prediction-of-Patients-with-Heart-Failure repository\n",
            "DATA_URL = \"https://raw.githubusercontent.com/dimikara/Survival-Prediction-of-Patients-with-Heart-Failure/master/heart_failure_clinical_records_dataset.csv\"\n",
            "\n",
            "print(f\"Ingesting data from: {DATA_URL}\")\n",
            "df = pd.read_csv(DATA_URL)\n",
            "print(f\"Dataset successfully loaded. Shape: {df.shape}\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display the first 5 records of the ingested dataset\n",
            "df.head()"
        ]
    })
    
    # -------------------------------------------------------------
    # 2. Data Preprocessing & Cleaning
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Preprocessing & Cleaning\n",
            "Handling missing values, removing duplicates, and normalizing data to make it usable."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Check for missing values\n",
            "print(\"--- Missing Values per Column ---\")\n",
            "print(df.isnull().sum())\n",
            "\n",
            "# 2. Check for duplicate rows\n",
            "duplicates = df.duplicated().sum()\n",
            "print(f\"\\nNumber of duplicate rows: {duplicates}\")\n",
            "\n",
            "# 3. Verify column data types\n",
            "print(\"\\n--- Column Data Types ---\")\n",
            "print(df.dtypes)"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Basic Outlier Detection (using Tukey's IQR Method for key numeric columns)\n",
            "numeric_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']\n",
            "\n",
            "print(\"Outlier Detection (Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]):\")\n",
            "for col in numeric_cols:\n",
            "    Q1 = df[col].quantile(0.25)\n",
            "    Q3 = df[col].quantile(0.75)\n",
            "    IQR = Q3 - Q1\n",
            "    lower_bound = Q1 - 1.5 * IQR\n",
            "    upper_bound = Q3 + 1.5 * IQR\n",
            "    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]\n",
            "    print(f\" - {col}: {len(outliers)} outliers (range: [{df[col].min()}, {df[col].max()}])\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Pre-processing Note**: The dataset does not contain missing values or duplicate rows. Outliers (e.g. extreme values of Creatinine Phosphokinase and Serum Creatinine) are common in clinical records due to patients with acute kidney failure or severe cardiac stress. Rather than removing these vital clinical signals, we will keep them and leverage robust algorithms (like Decision Trees/Random Forests) and robust scaling methods."
        ]
    })
    
    # -------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Feature Engineering\n",
            "Selecting and creating the most relevant features to improve model performance.\n",
            "\n",
            "### Avoid Data Leakage:\n",
            "The `time` variable in the dataset represents the patient's **follow-up duration (days)**. Because survival duration is only known *after* the patient dies or completes the study, it cannot be used at baseline to predict whether a patient will die. We **must drop** the `time` feature to prevent severe data leakage.\n",
            "\n",
            "### Clinical Feature Creation:\n",
            "1. **`creatinine_sodium_ratio`**: Creatinine and Sodium levels are correlated with cardiac and renal dysfunction. High creatinine coupled with low sodium is a classical indicator of poor prognosis.\n",
            "2. **`is_elderly`**: Patients aged 65 or above are clinically classified in the high-risk elderly demographic."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create features\n",
            "df_engineered = df.copy()\n",
            "\n",
            "# 1. Creatinine to Sodium Ratio\n",
            "df_engineered['creatinine_sodium_ratio'] = df_engineered['serum_creatinine'] / df_engineered['serum_sodium']\n",
            "\n",
            "# 2. Age demographic binary feature (Elderly threshold of 65)\n",
            "df_engineered['is_elderly'] = (df_engineered['age'] >= 65).astype(int)\n",
            "\n",
            "# Drop the leakage column 'time'\n",
            "df_engineered = df_engineered.drop(columns=['time'])\n",
            "\n",
            "print(f\"Features after engineering (excluding 'time'):\\n{list(df_engineered.columns)}\")\n",
            "df_engineered.head()"
        ]
    })
    
    # -------------------------------------------------------------
    # 4. Data Splitting
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Data Splitting\n",
            "Dividing data into training, validation, and testing sets.\n",
            "\n",
            "Because the dataset has a moderate class imbalance (~68% survival, ~32% mortality), we perform a **stratified split** to ensure training and test sets maintain identical label proportions."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "# Separate features and target label\n",
            "X = df_engineered.drop(columns=['DEATH_EVENT'])\n",
            "y = df_engineered['DEATH_EVENT']\n",
            "\n",
            "# Split into Train and Test sets (80% train, 20% test)\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "print(f\"Train features shape: {X_train.shape}\")\n",
            "print(f\"Test features shape: {X_test.shape}\")\n",
            "print(f\"\\nTrain class distribution:\\n{y_train.value_counts(normalize=True)}\")\n",
            "print(f\"\\nTest class distribution:\\n{y_test.value_counts(normalize=True)}\")"
        ]
    })
    
    # -------------------------------------------------------------
    # 5. Model Training & Tuning
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Training & Tuning\n",
            "Choosing algorithms and optimizing hyperparameters (using tools like Grid Search).\n",
            "\n",
            "We build predictive classifiers using two distinct algorithms:\n",
            "1. **Random Forest Classifier** (robust tree ensemble)\n",
            "2. **XGBoost Classifier** (highly efficient gradient boosted trees)\n",
            "\n",
            "We set up hyperparameter tuning via **GridSearchCV** with 5-fold cross-validation, optimizing for **ROC AUC** to handle class imbalance."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from xgboost import XGBClassifier\n",
            "from sklearn.model_selection import GridSearchCV\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.pipeline import Pipeline\n",
            "\n",
            "# 1. Tune Random Forest\n",
            "rf_pipeline = Pipeline([\n",
            "    ('scaler', StandardScaler()),\n",
            "    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))\n",
            "])\n",
            "\n",
            "rf_param_grid = {\n",
            "    'rf__n_estimators': [50, 100, 200],\n",
            "    'rf__max_depth': [3, 5, 8, None],\n",
            "    'rf__min_samples_split': [2, 5, 10]\n",
            "}\n",
            "\n",
            "print(\"Tuning Random Forest...\")\n",
            "rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)\n",
            "rf_grid.fit(X_train, y_train)\n",
            "\n",
            "print(f\"Best Random Forest Parameters: {rf_grid.best_params_}\")\n",
            "print(f\"Best Random Forest CV ROC AUC: {rf_grid.best_score_:.4f}\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate class ratio to adjust XGBoost scale_pos_weight\n",
            "ratio = (len(y_train) - sum(y_train)) / sum(y_train)\n",
            "\n",
            "# 2. Tune XGBoost\n",
            "xgb_pipeline = Pipeline([\n",
            "    ('scaler', StandardScaler()),\n",
            "    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=ratio))\n",
            "])\n",
            "\n",
            "xgb_param_grid = {\n",
            "    'xgb__n_estimators': [50, 100, 150],\n",
            "    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],\n",
            "    'xgb__max_depth': [3, 4, 5, 6]\n",
            "}\n",
            "\n",
            "print(\"Tuning XGBoost...\")\n",
            "xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)\n",
            "xgb_grid.fit(X_train, y_train)\n",
            "\n",
            "print(f\"Best XGBoost Parameters: {xgb_grid.best_params_}\")\n",
            "print(f\"Best XGBoost CV ROC AUC: {xgb_grid.best_score_:.4f}\")"
        ]
    })
    
    # -------------------------------------------------------------
    # 6. Model Evaluation
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Model Evaluation\n",
            "Validating the model's accuracy, precision, and recall against testing data.\n",
            "\n",
            "We compare both tuned models on the held-out test set using ROC AUC, confusion matrices, and detailed classification reports."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "best_rf = rf_grid.best_estimator_\n",
            "best_xgb = xgb_grid.best_estimator_\n",
            "\n",
            "models = {\n",
            "    \"Tuned Random Forest\": best_rf,\n",
            "    \"Tuned XGBoost\": best_xgb\n",
            "}\n",
            "\n",
            "for name, model in models.items():\n",
            "    preds = model.predict(X_test)\n",
            "    probs = model.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    print(\"=\" * 50)\n",
            "    print(f\"EVALUATING: {name}\")\n",
            "    print(\"=\" * 50)\n",
            "    print(f\"Test ROC AUC: {roc_auc_score(y_test, probs):.4f}\")\n",
            "    print(\"\\nClassification Report:\")\n",
            "    print(classification_report(y_test, preds))\n",
            "    print(\"\\n\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot Confusion Matrices and ROC Curves side-by-side\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 12))\n",
            "\n",
            "# Confusion Matrices\n",
            "ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, ax=axes[0, 0], cmap='Blues')\n",
            "axes[0, 0].set_title(\"Random Forest Confusion Matrix\")\n",
            "\n",
            "ConfusionMatrixDisplay.from_estimator(best_xgb, X_test, y_test, ax=axes[0, 1], cmap='Purples')\n",
            "axes[0, 1].set_title(\"XGBoost Confusion Matrix\")\n",
            "\n",
            "# ROC Curves\n",
            "RocCurveDisplay.from_estimator(best_rf, X_test, y_test, ax=axes[1, 0])\n",
            "axes[1, 0].plot([0, 1], [0, 1], 'r--')\n",
            "axes[1, 0].set_title(\"Random Forest ROC Curve\")\n",
            "\n",
            "RocCurveDisplay.from_estimator(best_xgb, X_test, y_test, ax=axes[1, 1])\n",
            "axes[1, 1].plot([0, 1], [0, 1], 'r--')\n",
            "axes[1, 1].set_title(\"XGBoost ROC Curve\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot Feature Importance for XGBoost\n",
            "xgb_model = best_xgb.named_steps['xgb']\n",
            "importances = xgb_model.feature_importances_\n",
            "indices = np.argsort(importances)[::-1]\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.title(\"XGBoost Feature Importances (Clinical Covariates Only)\", fontsize=14, fontweight='bold')\n",
            "plt.bar(range(X.shape[1]), importances[indices], align=\"center\", color='purple', alpha=0.7)\n",
            "plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')\n",
            "plt.ylabel(\"Relative Importance\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # -------------------------------------------------------------
    # 7. Deployment & Monitoring
    # -------------------------------------------------------------
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Deployment & Monitoring\n",
            "Deploying the model to production (using Flask, FastAPI, or cloud services) and monitoring for performance decay (model drift).\n",
            "\n",
            "### 7.1. Model Export\n",
            "First, we serialize our complete preprocessing and classification pipeline."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import joblib\n",
            "\n",
            "# Export the best performing pipeline (e.g., Tuned Random Forest or XGBoost)\n",
            "model_filename = 'heart_failure_xgb_pipeline.joblib'\n",
            "joblib.dump(best_xgb, model_filename)\n",
            "print(f\"Serialized pipeline saved to: {model_filename}\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 7.2. Serving API Code (FastAPI)\n",
            "Below is a complete FastAPI serving application that can load the saved pipeline and process inference requests. You can deploy this on platform services (e.g., AWS Elastic Beanstalk, Heroku, or GCP Run)."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Write FastAPI code structure as a template string (demonstrating FastAPI setup)\n",
            "fastapi_app_code = \"\"\"\\\n",
            "from fastapi import FastAPI, HTTPException\n",
            "from pydantic import BaseModel, Field\n",
            "import joblib\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Initialize FastAPI app\n",
            "app = FastAPI(\n",
            "    title=\"Heart Failure Mortality Predictor API\",\n",
            "    description=\"API serving a machine learning classifier to predict patient mortality risk based on clinical parameters.\",\n",
            "    version=\"1.0.0\"\n",
            ")\n",
            "\n",
            "# Load model pipeline\n",
            "try:\n",
            "    pipeline = joblib.load('heart_failure_xgb_pipeline.joblib')\n",
            "except Exception as e:\n",
            "    raise RuntimeError(f\"Failed to load serialized pipeline: {e}\")\n",
            "\n",
            "# Define schema for incoming inference request\n",
            "class PatientProfile(BaseModel):\n",
            "    age: float = Field(..., example=65.0, description=\"Age of the patient in years\")\n",
            "    anaemia: int = Field(..., example=0, description=\"Anaemia flag (0 or 1)\")\n",
            "    creatinine_phosphokinase: int = Field(..., example=250, description=\"Level of CPK enzyme in blood (mcg/L)\")\n",
            "    diabetes: int = Field(..., example=0, description=\"Diabetes flag (0 or 1)\")\n",
            "    ejection_fraction: int = Field(..., example=35, description=\"Ejection fraction percentage\")\n",
            "    high_blood_pressure: int = Field(..., example=1, description=\"Hypertension flag (0 or 1)\")\n",
            "    platelets: float = Field(..., example=263000.0, description=\"Platelets count in blood (kiloplatelets/mL)\")\n",
            "    serum_creatinine: float = Field(..., example=1.2, description=\"Serum creatinine level (mg/dL)\")\n",
            "    serum_sodium: int = Field(..., example=137, description=\"Serum sodium level (mEq/L)\")\n",
            "    sex: int = Field(..., example=1, description=\"Gender (0 for Female, 1 for Male)\")\n",
            "    smoking: int = Field(..., example=0, description=\"Smoking status (0 or 1)\")\n",
            "\n",
            "# Endpoint for healthcheck\n",
            "@app.get(\"/\")\n",
            "def read_root():\n",
            "    return {\"status\": \"healthy\", \"model\": \"heart_failure_xgb_pipeline\"}\n",
            "\n",
            "# Endpoint for predictions\n",
            "@app.post(\"/predict\")\n",
            "def predict_mortality(profile: PatientProfile):\n",
            "    # Convert JSON body to pandas dataframe\n",
            "    input_dict = profile.dict()\n",
            "    \n",
            "    # Re-create engineered features expected by the pipeline\n",
            "    input_dict['creatinine_sodium_ratio'] = input_dict['serum_creatinine'] / input_dict['serum_sodium']\n",
            "    input_dict['is_elderly'] = 1 if input_dict['age'] >= 65 else 0\n",
            "    \n",
            "    # Convert to DataFrame (ensuring column ordering matches training stage)\n",
            "    columns_order = [\n",
            "        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', \n",
            "        'ejection_fraction', 'high_blood_pressure', 'platelets', \n",
            "        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', \n",
            "        'creatinine_sodium_ratio', 'is_elderly'\n",
            "    ]\n",
            "    \n",
            "    df_patient = pd.DataFrame([input_dict])[columns_order]\n",
            "    \n",
            "    # Inference\n",
            "    prediction = int(pipeline.predict(df_patient)[0])\n",
            "    probability = float(pipeline.predict_proba(df_patient)[0][1])\n",
            "    \n",
            "    return {\n",
            "        \"mortality_prediction\": prediction,\n",
            "        \"mortality_probability\": round(probability, 4),\n",
            "        \"risk_classification\": \"HIGH_RISK\" if probability >= 0.5 else \"STANDARD_RISK\"\n",
            "    }\n",
            "\"\"\"\n",
            "\n",
            "print(\"FastAPI serving script template ready for deployment:\")\n",
            "print(fastapi_app_code[:400] + \"\\n... [truncated for display] ...\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 7.3. Model Drift & Production Monitoring\n",
            "\n",
            "Once a clinical predictive model is deployed in a production setting, its performance will degrade over time due to shifts in patient populations or clinical standards. This is referred to as **Model Drift**.\n",
            "\n",
            "#### Two Main Types of Drift:\n",
            "1. **Covariate Shift (Data Drift)**: The distribution of input features ($P(X)$) changes over time. For example, a hospital might begin treating older patients on average, shifting the distribution of the `age` feature.\n",
            "2. **Concept Drift**: The relationship between inputs and the target label ($P(Y|X)$) changes. For example, a new drug or treatment standard might drastically decrease mortality rate for patients who have high serum creatinine, rendering the original model's predictions obsolete.\n",
            "\n",
            "#### Implementing a Monitoring Strategy:\n",
            "- **Data Quality Auditing**: Monitor input statistical distributions (Min, Max, Mean, Variance) weekly. Use statistical tests like the **Kolmogorov-Smirnov (KS) test** or **Population Stability Index (PSI)** to compare production data against training benchmarks.\n",
            "- **Metric Logging**: Log patient inputs, predicted risk probabilities, and actual clinical outcomes (ground truth labels) when they become available. Monitor sliding window accuracy, ROC AUC, and recall.\n",
            "- **Alerting**: Set up automatic alerts if PSI exceeds 0.2 (indicating significant drift) or if model test ROC AUC drops below 0.70.\n",
            "- **Triggered Retraining**: Schedule retraining cycles when drift is detected. Collect new labeled patient records, append them to the training set, and execute hyperparameter re-tuning."
        ]
    })
    
    with open('heart_failure_mortality_predictor.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Notebook generated successfully!")

if __name__ == '__main__':
    create_notebook()

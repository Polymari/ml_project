import json

def append_project_b():
    with open('heart_failure_mortality_predictor.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Check if Project B cells are already appended (to prevent double appending)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown' and 'Project B: Survival Modeling' in ''.join(cell['source']):
            print("Project B cells are already appended.")
            return

    new_cells = [
        # --- Heading ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "\n",
                "---\n",
                "\n",
                "# Project B: Survival Modeling & Risk Profiling (Survival Analysis)\n",
                "\n",
                "Unlike classification (where we predict *whether* an event will happen and must drop `time` to prevent leakage), **Survival Analysis** models *when* the event will happen. It accounts for **censoring** (patients who did not die during the follow-up period).\n",
                "\n",
                "In this section, we build a survival analysis pipeline using the **Cox Proportional Hazards (CoxPH)** model to predict individual patient survival curves over time."
            ]
        },
        # --- Install lifelines ---
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install lifelines for survival analysis\n",
                "!pip install -q lifelines"
            ]
        },
        # --- Imports ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B.1. Setup & Data Preparation\n",
                "For survival analysis, our target includes both the duration (`time`) and the event indicator (`DEATH_EVENT`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from lifelines import CoxPHFitter\n",
                "from lifelines.utils import concordance_index\n",
                "from sklearn.model_selection import train_test_split\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Load raw dataset\n",
                "df_survival = pd.read_csv(DATA_URL)\n",
                "\n",
                "# Add engineered features from Project A\n",
                "df_survival['creatinine_sodium_ratio'] = df_survival['serum_creatinine'] / df_survival['serum_sodium']\n",
                "df_survival['is_elderly'] = (df_survival['age'] >= 65).astype(int)\n",
                "\n",
                "print(f\"Survival dataset shape: {df_survival.shape}\")"
            ]
        },
        # --- Data Splitting ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B.2. Train / Test Split for Survival Data\n",
                "We split our data into training (80%) and testing (20%) sets. We stratify by `DEATH_EVENT` to ensure equal proportion of event occurrences in both splits."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_df, test_df = train_test_split(\n",
                "    df_survival, test_size=0.2, random_state=42, stratify=df_survival['DEATH_EVENT']\n",
                ")\n",
                "\n",
                "print(f\"Train samples: {len(train_df)}\")\n",
                "print(f\"Test samples: {len(test_df)}\")"
            ]
        },
        # --- Model Fitting ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B.3. Training Cox Proportional Hazards Model\n",
                "We fit the CoxPH model on the training dataset. We specify `time` as our duration column and `DEATH_EVENT` as our event column."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cph = CoxPHFitter(penalizer=0.1)  # Penalization helps prevent overfitting / high coefficients\n",
                "cph.fit(train_df, duration_col='time', event_col='DEATH_EVENT')\n",
                "\n",
                "# Display statistical summary\n",
                "cph.print_summary()"
            ]
        },
        # --- Model Evaluation ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B.4. Model Evaluation (Concordance Index)\n",
                "The **Concordance Index (C-index)** measures the model's ability to correctly order survival times. A C-index of 0.5 indicates random predictions, while 1.0 indicates perfect ordering."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate on training set\n",
                "train_c_index = concordance_index(\n",
                "    train_df['time'], \n",
                "    -cph.predict_partial_hazard(train_df), \n",
                "    train_df['DEATH_EVENT']\n",
                ")\n",
                "\n",
                "# Evaluate on testing set\n",
                "test_c_index = concordance_index(\n",
                "    test_df['time'], \n",
                "    -cph.predict_partial_hazard(test_df), \n",
                "    test_df['DEATH_EVENT']\n",
                ")\n",
                "\n",
                "print(f\"Train Concordance Index: {train_c_index:.4f}\")\n",
                "print(f\"Test Concordance Index: {test_c_index:.4f}\")"
            ]
        },
        # --- Individual Profiling ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B.5. Individual Patient Risk Profiling\n",
                "One of the major strengths of survival analysis is predicting **individualized survival curves** over time.\n",
                "\n",
                "Let's select two distinct patients from our test set:\n",
                "1. **Patient A**: Younger patient with standard ejection fraction and serum creatinine levels.\n",
                "2. **Patient B**: Elderly patient with low ejection fraction and high serum creatinine levels.\n",
                "\n",
                "We will compare their predicted survival probabilities over the 250+ day monitoring timeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Identify a low-risk and high-risk patient in test_df\n",
                "# Low risk: low serum creatinine (<1.0), high ejection fraction (>45), younger age (<55)\n",
                "low_risk_candidates = test_df[\n",
                "    (test_df['serum_creatinine'] < 1.0) & \n",
                "    (test_df['ejection_fraction'] > 45) & \n",
                "    (test_df['age'] < 55)\n",
                "]\n",
                "\n",
                "# High risk: high serum creatinine (>2.0), low ejection fraction (<30), older age (>70)\n",
                "high_risk_candidates = test_df[\n",
                "    (test_df['serum_creatinine'] > 1.8) & \n",
                "    (test_df['ejection_fraction'] < 30) & \n",
                "    (test_df['age'] > 70)\n",
                "]\n",
                "\n",
                "print(f\"Low-risk candidates found: {len(low_risk_candidates)}\")\n",
                "print(f\"High-risk candidates found: {len(high_risk_candidates)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Select one patient from each group\n",
                "patient_low = low_risk_candidates.iloc[[0]].drop(columns=['time', 'DEATH_EVENT'])\n",
                "patient_high = high_risk_candidates.iloc[[0]].drop(columns=['time', 'DEATH_EVENT'])\n",
                "\n",
                "print(\"--- Low-Risk Patient Profile ---\")\n",
                "print(patient_low.to_string(index=False))\n",
                "\n",
                "print(\"\\n--- High-Risk Patient Profile ---\")\n",
                "print(patient_high.to_string(index=False))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate survival curves\n",
                "surv_low = cph.predict_survival_function(patient_low)\n",
                "surv_high = cph.predict_survival_function(patient_high)\n",
                "\n",
                "# Plot curves\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.plot(surv_low.index, surv_low.values, label='Patient A: Low Risk Profile', color='green', linewidth=2.5)\n",
                "plt.plot(surv_high.index, surv_high.values, label='Patient B: High Risk Profile', color='red', linewidth=2.5)\n",
                "\n",
                "plt.title('Predicted Individual Patient Survival Probabilities Over Time', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Time (Days)', fontsize=12)\n",
                "plt.ylabel('Survival Probability', fontsize=12)\n",
                "plt.ylim(0, 1.05)\n",
                "plt.legend(fontsize=11)\n",
                "plt.grid(True, linestyle='--', alpha=0.6)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Print 30-day, 90-day, and 180-day survival probability comparisons\n",
                "days = [30, 90, 180]\n",
                "print(\"Predicted Survival Probabilities Comparison:\")\n",
                "print(\"-\" * 55)\n",
                "for day in days:\n",
                "    # Find the index closest to the day\n",
                "    idx_low = surv_low.index.get_indexer([day], method='nearest')[0]\n",
                "    idx_high = surv_high.index.get_indexer([day], method='nearest')[0]\n",
                "    \n",
                "    prob_low = surv_low.iloc[idx_low].values[0]\n",
                "    prob_high = surv_high.iloc[idx_high].values[0]\n",
                "    \n",
                "    print(f\"{day}-Day Survival Prob: Patient A = {prob_low:.2%}, Patient B = {prob_high:.2%}\")"
            ]
        }
    ]
    
    notebook['cells'].extend(new_cells)
    
    with open('heart_failure_mortality_predictor.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Project B cells appended successfully!")

if __name__ == '__main__':
    append_project_b()

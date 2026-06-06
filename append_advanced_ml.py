import json

def append_advanced():
    with open('heart_failure_mortality_predictor.ipynb', 'r') as f:
        notebook = json.load(f)
        
    # Check if Project C is already appended
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown' and 'Project C: Model Interpretability' in ''.join(cell['source']):
            print("Project C/D cells are already appended.")
            return

    new_cells = [
        # --- Project C Heading ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "\n",
                "---\n",
                "\n",
                "# Project C: Model Interpretability with SHAP\n",
                "\n",
                "Clinical models must be transparent. We use **SHAP (SHapley Additive exPlanations)** to explain individual patient risk predictions from the trained XGBoost model. This reveals which clinical features drove a patient's risk up or down."
            ]
        },
        # --- Install SHAP ---
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q shap"
            ]
        },
        # --- SHAP Calculations ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### C.1. Compute SHAP Values\n",
                "We extract the trained XGBoost model from the pipeline and use a TreeExplainer to calculate SHAP values for the test set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shap\n",
                "\n",
                "xgb_model = best_xgb.named_steps['xgb']\n",
                "scaler = best_xgb.named_steps['scaler']\n",
                "\n",
                "X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)\n",
                "\n",
                "explainer = shap.TreeExplainer(xgb_model)\n",
                "shap_values = explainer(X_test_scaled)\n",
                "\n",
                "print(\"SHAP values calculated successfully. Shape:\", shap_values.shape)"
            ]
        },
        # --- SHAP Summary Plot ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### C.2. Global Feature Attribution (Summary Plot)\n",
                "The summary plot displays how high or low values of a feature impact the mortality prediction across all test patients."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "shap.summary_plot(shap_values, X_test_scaled, show=False)\n",
                "plt.title(\"SHAP Global Feature Importance & Directional Impact\", fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # --- SHAP Individual Prediction Plot ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### C.3. Local Patient Attribution (Waterfall Plot)\n",
                "We generate a waterfall plot for the first patient in the test set to explain why they were classified as high or standard risk."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "shap.plots.waterfall(shap_values[0], show=False)\n",
                "plt.title(\"Patient-Specific SHAP Risk Attribution\", fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # --- Project D Heading ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "\n",
                "---\n",
                "\n",
                "# Project D: Patient Phenotyping via Unsupervised Clustering\n",
                "\n",
                "We use unsupervised learning to discover patient subgroups (phenogroups) based on continuous clinical features. This helps identify patient cohorts with distinct medical profiles and risk profiles."
            ]
        },
        # --- Clustering Code ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### D.1. Standardize and Cluster Patients\n",
                "We extract all continuous features, scale them, and apply **K-Means clustering** to partition patients into 3 clinical groups."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.cluster import KMeans\n",
                "from sklearn.decomposition import PCA\n",
                "\n",
                "continuous_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']\n",
                "X_continuous = df_survival[continuous_cols]\n",
                "\n",
                "scaler_cluster = StandardScaler()\n",
                "X_continuous_scaled = scaler_cluster.fit_transform(X_continuous)\n",
                "\n",
                "kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n",
                "cluster_labels = kmeans.fit_predict(X_continuous_scaled)\n",
                "\n",
                "df_clusters = df_survival.copy()\n",
                "df_clusters['Cluster'] = cluster_labels\n",
                "\n",
                "print(\"Cluster assignments count:\")\n",
                "print(df_clusters['Cluster'].value_counts())"
            ]
        },
        # --- PCA Visualization ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### D.2. Visualizing Clusters via PCA\n",
                "We project our continuous features into 2D space using Principal Component Analysis (PCA) to visually inspect cluster separation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pca = PCA(n_components=2, random_state=42)\n",
                "X_pca = pca.fit_transform(X_continuous_scaled)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.scatterplot(\n",
                "    x=X_pca[:, 0], y=X_pca[:, 1], \n",
                "    hue=df_clusters['Cluster'], \n",
                "    palette='Set1', alpha=0.8, s=80\n",
                ")\n",
                "plt.title(\"Patient Clustering Visualized via PCA\", fontsize=14, fontweight='bold')\n",
                "plt.xlabel(\"Principal Component 1\")\n",
                "plt.ylabel(\"Principal Component 2\")\n",
                "plt.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # --- Cluster Profiling ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### D.3. Cluster Profiling\n",
                "We group the dataset by cluster labels and compute mean features to clinically define each patient phenotype and check their actual mortality rates."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "profile_cols = continuous_cols + ['DEATH_EVENT']\n",
                "cluster_profile = df_clusters.groupby('Cluster')[profile_cols].mean()\n",
                "print(cluster_profile.to_string())"
            ]
        }
    ]
    
    notebook['cells'].extend(new_cells)
    
    with open('heart_failure_mortality_predictor.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Project C & D cells appended successfully!")

if __name__ == '__main__':
    append_advanced()

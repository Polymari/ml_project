import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from src.features import add_features

xgb_pipeline = joblib.load('heart_failure_xgb_pipeline.joblib')
cph_model = joblib.load('heart_failure_cph_model.joblib')
kmeans_artifact = joblib.load('heart_failure_kmeans.joblib')

xgb_model = xgb_pipeline.named_steps['xgb']
xgb_scaler = xgb_pipeline.named_steps['scaler']

kmeans_model = kmeans_artifact['model']
kmeans_scaler = kmeans_artifact['scaler']
kmeans_features = kmeans_artifact['features']
cluster_profiles = kmeans_artifact['profiles']

explainer = shap.TreeExplainer(xgb_model)

PHENOTYPE_NAMES = {
    0: "Phenotype A",
    1: "Phenotype B",
    2: "Phenotype C",
}

PHENOTYPE_COLORS = {
    0: "#6366F1",
    1: "#F59E0B",
    2: "#EC4899",
}


def describe_phenotype(cluster_id, profiles):
    """Generates a human-readable clinical description for a cluster."""
    row = profiles.loc[cluster_id]
    mortality_pct = row['DEATH_EVENT'] * 100

    age_desc = "elderly" if row['age'] >= 65 else "middle-aged" if row['age'] >= 50 else "younger"
    ef_desc = "reduced" if row['ejection_fraction'] < 35 else "mildly reduced" if row['ejection_fraction'] < 45 else "preserved"
    cr_desc = "elevated" if row['serum_creatinine'] > 1.5 else "borderline" if row['serum_creatinine'] > 1.2 else "normal"

    return (
        f"Predominantly {age_desc} patients (avg age {row['age']:.0f}) with "
        f"{ef_desc} ejection fraction ({row['ejection_fraction']:.0f}%) and "
        f"{cr_desc} serum creatinine ({row['serum_creatinine']:.1f} mg/dL). "
        f"Cohort mortality rate: {mortality_pct:.0f}%."
    )


def predict_and_explain(
    age, anaemia, cpk, diabetes, ejection_fraction,
    high_blood_pressure, platelets, serum_creatinine,
    serum_sodium, sex, smoking
):
    anaemia_val = 1 if anaemia == "Yes" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    hbp_val = 1 if high_blood_pressure == "Yes" else 0
    sex_val = 1 if sex == "Male" else 0
    smoking_val = 1 if smoking == "Yes" else 0

    patient_dict = {
        'age': float(age),
        'anaemia': int(anaemia_val),
        'creatinine_phosphokinase': int(cpk),
        'diabetes': int(diabetes_val),
        'ejection_fraction': int(ejection_fraction),
        'high_blood_pressure': int(hbp_val),
        'platelets': float(platelets),
        'serum_creatinine': float(serum_creatinine),
        'serum_sodium': int(serum_sodium),
        'sex': int(sex_val),
        'smoking': int(smoking_val)
    }

    df_patient = pd.DataFrame([patient_dict])
    df_patient = add_features(df_patient)

    prob = float(xgb_pipeline.predict_proba(df_patient)[0][1])

    risk_label = "HIGH RISK" if prob >= 0.5 else "STANDARD RISK"
    color = "#FF4B4B" if prob >= 0.5 else "#2E7D32"

    risk_md = f"<div style='text-align: center; padding: 15px; border-radius: 8px; background-color: {color}20; border: 2px solid {color};'>"
    risk_md += f"<p style='margin: 0; font-size: 14px; font-weight: bold; color: {color};'>CLASSIFICATION</p>"
    risk_md += f"<h2 style='margin: 5px 0 0 0; font-size: 28px; font-weight: 900; color: {color};'>{risk_label}</h2>"
    risk_md += f"<p style='margin: 5px 0 0 0; font-size: 16px; color: {color};'>Calculated Risk Probability: {prob:.2%}</p>"
    risk_md += "</div>"

    patient_continuous = df_patient[kmeans_features].values
    patient_scaled_cluster = kmeans_scaler.transform(patient_continuous)
    cluster_id = int(kmeans_model.predict(patient_scaled_cluster)[0])

    pheno_name = PHENOTYPE_NAMES[cluster_id]
    pheno_color = PHENOTYPE_COLORS[cluster_id]
    pheno_desc = describe_phenotype(cluster_id, cluster_profiles)

    pheno_md = f"<div style='padding: 15px; border-radius: 8px; background-color: {pheno_color}15; border: 2px solid {pheno_color};'>"
    pheno_md += f"<p style='margin: 0; font-size: 14px; font-weight: bold; color: {pheno_color};'>PATIENT PHENOTYPE</p>"
    pheno_md += f"<h2 style='margin: 5px 0 0 0; font-size: 24px; font-weight: 900; color: {pheno_color};'>{pheno_name}</h2>"
    pheno_md += f"<p style='margin: 8px 0 0 0; font-size: 13px; color: #D1D5DB; line-height: 1.5;'>{pheno_desc}</p>"
    pheno_md += "<div style='margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;'>"

    profile_row = cluster_profiles.loc[cluster_id]
    labels = {
        'age': 'Avg Age', 'ejection_fraction': 'Avg EF%',
        'serum_creatinine': 'Avg Creatinine', 'serum_sodium': 'Avg Sodium',
        'creatinine_phosphokinase': 'Avg CPK', 'platelets': 'Avg Platelets'
    }
    for feat in kmeans_features:
        val = profile_row[feat]
        fmt = f"{val:,.0f}" if feat in ('platelets', 'creatinine_phosphokinase') else f"{val:.1f}"
        pheno_md += f"<span style='background: {pheno_color}30; color: {pheno_color}; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;'>{labels[feat]}: {fmt}</span>"

    pheno_md += "</div></div>"

    patient_scaled_xgb = pd.DataFrame(xgb_scaler.transform(df_patient), columns=df_patient.columns)
    shap_vals = explainer(patient_scaled_xgb)

    plt.close('all')
    shap.plots.waterfall(shap_vals[0], show=False)
    fig_shap = plt.gcf()
    fig_shap.set_size_inches(9, 4.5)
    plt.tight_layout()

    patient_cph_df = df_patient.copy()
    surv = cph_model.predict_survival_function(patient_cph_df)

    plt.figure(figsize=(9, 4.5))
    fig_surv = plt.gcf()
    plt.plot(surv.index, surv.values, color='dodgerblue', linewidth=2.5)
    plt.title("Predicted Patient Survival Probability Over Time", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Follow-up Duration (Days)", fontsize=10)
    plt.ylabel("Survival Probability", fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    return risk_md, pheno_md, fig_surv, fig_shap

app = gr.Blocks()

with app:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 25px;">
            <h1 style="font-weight: 900; font-size: 32px; color: #1E3A8A; margin-bottom: 5px;">Heart Failure Mortality & Survival Predictor</h1>
            <p style="font-size: 16px; color: #4B5563; margin-top: 0;">Dual ML Pipeline: XGBoost Risk Classification & Cox Proportional Hazards Survival Curves</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
                <div style="background-color: #F3F4F6; padding: 10px 15px; border-radius: 6px; border-left: 4px solid #1E3A8A; margin-bottom: 15px;">
                    <h3 style="margin: 0; font-size: 16px; font-weight: bold; color: #1E3A8A;">Patient Clinical Profile</h3>
                </div>
            """)

            age = gr.Slider(minimum=40, maximum=100, value=65, step=1, label="Age (Years)")

            with gr.Row():
                sex = gr.Dropdown(choices=["Female", "Male"], value="Male", label="Sex")
                smoking = gr.Dropdown(choices=["No", "Yes"], value="No", label="Smoking Status")

            with gr.Row():
                anaemia = gr.Dropdown(choices=["No", "Yes"], value="No", label="Anaemia")
                diabetes = gr.Dropdown(choices=["No", "Yes"], value="No", label="Diabetes")
                high_blood_pressure = gr.Dropdown(choices=["No", "Yes"], value="No", label="Hypertension")

            ejection_fraction = gr.Slider(minimum=10, maximum=80, value=35, step=1, label="Ejection Fraction (%)")
            serum_creatinine = gr.Slider(minimum=0.5, maximum=10.0, value=1.2, step=0.1, label="Serum Creatinine (mg/dL)")
            serum_sodium = gr.Slider(minimum=110, maximum=150, value=137, step=1, label="Serum Sodium (mEq/L)")

            with gr.Row():
                cpk = gr.Number(value=250, label="CPK Enzyme Level (mcg/L)", precision=0)
                platelets = gr.Number(value=263000, label="Platelets Count (kiloplatelets/mL)", precision=0)

            predict_btn = gr.Button("Evaluate Patient Risk Profile", variant="primary")

        with gr.Column(scale=2):
            gr.HTML("""
                <div style="background-color: #F3F4F6; padding: 10px 15px; border-radius: 6px; border-left: 4px solid #10B981; margin-bottom: 15px;">
                    <h3 style="margin: 0; font-size: 16px; font-weight: bold; color: #065F46;">Diagnostic Results</h3>
                </div>
            """)

            output_risk = gr.HTML(value="<div style='text-align: center; color: #6B7280; padding: 20px;'>Enter patient parameters on the left and click 'Evaluate' to run diagnostic pipeline.</div>")
            output_pheno = gr.HTML()
            output_surv = gr.Plot(label="Survival Probability Curve")
            output_shap = gr.Plot(label="SHAP Risk Attribution")

    predict_btn.click(
        fn=predict_and_explain,
        inputs=[
            age, anaemia, cpk, diabetes, ejection_fraction,
            high_blood_pressure, platelets, serum_creatinine,
            serum_sodium, sex, smoking
        ],
        outputs=[output_risk, output_pheno, output_surv, output_shap]
    )

if __name__ == '__main__':
    app.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))

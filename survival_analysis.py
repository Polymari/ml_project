import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
import os
import shutil

def run_survival():
    # Load dataset
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    # 1. Fit Cox Proportional Hazards Model
    cph = CoxPHFitter()
    # We will fit on all clinical variables, using 'time' as duration and 'DEATH_EVENT' as event
    cph.fit(df, duration_col='time', event_col='DEATH_EVENT')
    
    print("==================================================")
    print("COX PROPORTIONAL HAZARDS MODEL SUMMARY")
    print("==================================================")
    print(cph.summary.to_string())
    
    # Create plots directory if not exists
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 2. Kaplan-Meier Survival Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Curve A: Ejection Fraction (High vs Low, split by median 38)
    kmf = KaplanMeierFitter()
    median_ef = df['ejection_fraction'].median()
    high_ef = df['ejection_fraction'] >= median_ef
    
    kmf.fit(df.loc[high_ef, 'time'], df.loc[high_ef, 'DEATH_EVENT'], label=f'High Ejection Fraction (>= {median_ef}%)')
    kmf.plot(ax=axes[0])
    
    kmf.fit(df.loc[~high_ef, 'time'], df.loc[~high_ef, 'DEATH_EVENT'], label=f'Low Ejection Fraction (< {median_ef}%)')
    kmf.plot(ax=axes[0])
    
    axes[0].set_title('Survival Curve by Ejection Fraction Group', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Follow-up Time (Days)')
    axes[0].set_ylabel('Survival Probability')
    axes[0].set_ylim(0, 1.05)
    
    # Curve B: Serum Creatinine (High vs Normal, split by median 1.1)
    median_sc = df['serum_creatinine'].median()
    high_sc = df['serum_creatinine'] >= median_sc
    
    kmf.fit(df.loc[high_sc, 'time'], df.loc[high_sc, 'DEATH_EVENT'], label=f'High Serum Creatinine (>= {median_sc} mg/dL)')
    kmf.plot(ax=axes[1])
    
    kmf.fit(df.loc[~high_sc, 'time'], df.loc[~high_sc, 'DEATH_EVENT'], label=f'Low Serum Creatinine (< {median_sc} mg/dL)')
    kmf.plot(ax=axes[1])
    
    axes[1].set_title('Survival Curve by Serum Creatinine Group', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Follow-up Time (Days)')
    axes[1].set_ylabel('Survival Probability')
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('plots/survival_curves.png', dpi=150)
    plt.close()
    
    # Copy to artifacts directory
    artifact_dir = r'C:\Users\gioha\.gemini\antigravity\brain\4b6f2aff-75cd-4e22-a92e-50ca0443ba06\plots'
    os.makedirs(artifact_dir, exist_ok=True)
    shutil.copy('plots/survival_curves.png', os.path.join(artifact_dir, 'survival_curves.png'))
    print("\nSurvival plots generated and copied to artifacts directory successfully!")

if __name__ == '__main__':
    run_survival()

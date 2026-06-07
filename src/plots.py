import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str = 'plots'):
    """Generates and saves a correlation heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix of Heart Failure Clinical Records", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
    plt.close()


def plot_distributions(df: pd.DataFrame, output_dir: str = 'plots'):
    """Generates and saves distribution plots for key features."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='DEATH_EVENT', kde=True, multiple='stack', palette='crest')
    plt.title('Age Distribution vs Death Event', fontsize=12, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='serum_creatinine', hue='DEATH_EVENT', kde=True, multiple='stack', palette='magma', log_scale=True)
    plt.title('Serum Creatinine Distribution (Log Scale) vs Death Event', fontsize=12, fontweight='bold')
    plt.xlabel('Serum Creatinine (mg/dL)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'serum_creatinine_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='ejection_fraction', hue='DEATH_EVENT', kde=True, multiple='stack', palette='flare')
    plt.title('Ejection Fraction Distribution vs Death Event', fontsize=12, fontweight='bold')
    plt.xlabel('Ejection Fraction (%)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ejection_fraction_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='ejection_fraction', y='serum_creatinine', hue='DEATH_EVENT', palette='Set1', alpha=0.8)
    plt.yscale('log')
    plt.title('Ejection Fraction vs Serum Creatinine (Log Scale)', fontsize=12, fontweight='bold')
    plt.xlabel('Ejection Fraction (%)')
    plt.ylabel('Serum Creatinine (mg/dL)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_ef_creatinine.png'), dpi=150)
    plt.close()


def plot_survival_curves(df: pd.DataFrame, output_dir: str = 'plots'):
    """Generates and saves Kaplan-Meier survival curves."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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
    plt.savefig(os.path.join(output_dir, 'survival_curves.png'), dpi=150)
    plt.close()

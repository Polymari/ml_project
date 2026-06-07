import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load dataset
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix of Heart Failure Clinical Records", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=150)
    plt.close()
    
    # 2. Distribution of Age vs Death Event
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='DEATH_EVENT', kde=True, multiple='stack', palette='crest')
    plt.title('Age Distribution vs Death Event', fontsize=12, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/age_distribution.png', dpi=150)
    plt.close()
    
    # 3. Distribution of Serum Creatinine vs Death Event
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='serum_creatinine', hue='DEATH_EVENT', kde=True, multiple='stack', palette='magma', log_scale=True)
    plt.title('Serum Creatinine Distribution (Log Scale) vs Death Event', fontsize=12, fontweight='bold')
    plt.xlabel('Serum Creatinine (mg/dL)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/serum_creatinine_distribution.png', dpi=150)
    plt.close()
    
    # 4. Distribution of Ejection Fraction vs Death Event
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='ejection_fraction', hue='DEATH_EVENT', kde=True, multiple='stack', palette='flare')
    plt.title('Ejection Fraction Distribution vs Death Event', fontsize=12, fontweight='bold')
    plt.xlabel('Ejection Fraction (%)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/ejection_fraction_distribution.png', dpi=150)
    plt.close()
    
    # 5. Scatter Plot: Ejection Fraction vs Serum Creatinine
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='ejection_fraction', y='serum_creatinine', hue='DEATH_EVENT', palette='Set1', alpha=0.8)
    plt.yscale('log')
    plt.title('Ejection Fraction vs Serum Creatinine (Log Scale)', fontsize=12, fontweight='bold')
    plt.xlabel('Ejection Fraction (%)')
    plt.ylabel('Serum Creatinine (mg/dL)')
    plt.tight_layout()
    plt.savefig('plots/scatter_ef_creatinine.png', dpi=150)
    plt.close()

    print("Plots generated successfully!")

if __name__ == '__main__':
    main()

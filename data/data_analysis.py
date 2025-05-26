import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.data_cleaning import num_cols
from data.data_cleaning import df_clean

# Set style for plots
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Target variable distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Churn', data=df_clean)
plt.title('Churn Distribution')

# Tenure distribution
plt.subplot(2, 2, 2)
sns.histplot(df_clean['tenure'], bins=30, kde=True)
plt.title('Tenure Distribution')

# Monthly Charges vs Churn
plt.subplot(2, 2, 3)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_clean)
plt.title('Monthly Charges by Churn Status')

# Churn by Contract Type
plt.subplot(2, 2, 4)
contract_churn = df_clean.groupby('Contract')['Churn'].mean().sort_values()
contract_churn.plot(kind='bar')
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')

plt.tight_layout()
plt.savefig('reports/figures/eda_plots.png')
plt.show()

# Correlation matrix for numerical features
corr_matrix = df_clean[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Numerical Features Correlation Matrix')
plt.savefig('reports/figures/correlation_matrix.png')
plt.show()

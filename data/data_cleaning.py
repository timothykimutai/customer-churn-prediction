import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('data/raw_data.csv')
# Create a copy for cleaning
df_clean = df.copy()

# Handle missing values
print("Missing values before cleaning:")
print(df_clean.isnull().sum())

# TotalCharges has some empty strings that should be NA
df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan).astype(float)
# Fill the missing values in 'TotalCharges' with the median
total_charges_median = df_clean['TotalCharges'].median()
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(total_charges_median)

# Convert SeniorCitizen to categorical
df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0:'No', 1:'Yes'})

# Drop CustomerID
df_clean.drop('customerID', axis=1, inplace=True)

# Convert churn to binary
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

# Check for outliers in numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\nNumerical features summary:")
print(df_clean[num_cols].describe())

# Save cleaned data
df_clean.to_csv('data/cleaned_churn_data.csv', index=False)

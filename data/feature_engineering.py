import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/cleaned_churn_data.csv')

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical and numerical columns
cat_cols = [col for col in X.columns if X[col].dtype == 'object']
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Create new features
def create_features(df):
    df= df.copy()
    # Tenure categories
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, np.inf],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])
    # Total charges to tenure ratio
    df['charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1) # +1 to avoid division by zero
    
    # Flag for customers with both streaming TV and movies
    df['streaming_both'] =  ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(int)
    
    return df
X_fe = create_features(X)

# Update categorical and numerical columns
cat_cols += ['tenure_group']
num_cols += ['charge_per_tenure', 'streaming_both']

# Save feature-engineered data
X_fe.to_csv('data/feature_engineered_data.csv', index=False)
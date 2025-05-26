import pandas as pd

# Load dataset from kaggle 
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# save raw data
df.to_csv('data/raw_data.csv', index=False)
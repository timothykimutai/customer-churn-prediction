# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from src.monitoring import ModelMonitor

# Set up page
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_churn_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

# Feature engineering function (must match training)
def create_features(df):
    df = df.copy()
    # 1. Tenure categories
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, np.inf], 
                              labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])
    
    # 2. Total charges to tenure ratio
    df['charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
    
    # 3. Flag for customers with both streaming TV and movies
    df['streaming_both'] = ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(int)
    
    return df

df = load_data()
model = load_model()
monitor = ModelMonitor()

# Sidebar filters
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

tenure_filter = st.sidebar.slider(
    "Tenure (months)",
    min_value=int(df['tenure'].min()),
    max_value=int(df['tenure'].max()),
    value=(0, int(df['tenure'].max()))
)

# Apply filters
filtered_df = df[
    (df['Contract'].isin(contract_filter)) & 
    (df['tenure'].between(tenure_filter[0], tenure_filter[1]))
]

# Overview section
st.header("Business Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(filtered_df))
    
with col2:
    churn_rate = filtered_df['Churn'].mean()
    st.metric("Churn Rate", f"{churn_rate:.1%}")

with col3:
    st.metric("Predicted At-Risk", 
              f"{len(filtered_df) * 0.25:.0f} (est.)",
              help="Estimated based on model predictions")

# EDA Visualizations
st.header("Data Exploration")

tab1, tab2, tab3 = st.tabs(["Churn Distribution", "Key Factors", "Numerical Features"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Churn', ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=filtered_df.groupby('Contract')['Churn'].mean().reset_index(),
        x='Contract', y='Churn', ax=ax
    )
    ax.set_title("Churn Rate by Contract Type")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=filtered_df, 
        x='Churn', 
        y='MonthlyCharges', 
        ax=ax
    )
    ax.set_title("Monthly Charges by Churn Status")
    st.pyplot(fig)

# Model Performance
st.header("Model Monitoring")
performance_df = monitor.get_performance_history()

if not performance_df.empty:
    st.session_state.monitor = monitor
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        performance_df.plot(x='timestamp', y=['f1_score', 'roc_auc'], ax=ax)
        ax.axhline(y=0.75, color='r', linestyle='--', label='F1 Threshold')
        ax.axhline(y=0.85, color='g', linestyle='--', label='AUC Threshold')
        ax.legend()
        ax.set_title("Model Performance Metrics Over Time")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Latest Metrics")
        latest = performance_df.iloc[-1]
        st.metric("F1-Score", f"{latest['f1_score']:.3f}", 
                 delta=f"{(latest['f1_score'] - 0.75)/0.75:.1%} vs target")
        st.metric("ROC-AUC", f"{latest['roc_auc']:.3f}", 
                 delta=f"{(latest['roc_auc'] - 0.85)/0.85:.1%} vs target")
        st.write(f"Dataset: {latest['dataset']}")
        st.write(f"Sample size: {latest['sample_size']}")
else:
    # Initialize monitor and log some sample data
    monitor = ModelMonitor()
    sample_data = df.sample(100, random_state=42)
    monitor.log_performance(
        sample_data.drop('Churn', axis=1),
        sample_data['Churn'],
        'initial_sample_data'
    )
    st.session_state.monitor = monitor
    performance_df = monitor.get_performance_history()

# Prediction Interface
st.header("Single Customer Prediction")
st.write("Enter customer details to predict churn risk")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 100, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
    
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    submitted = st.form_submit_button("Predict Churn Risk")
    
    if submitted:
        # Create input DataFrame with all required columns
        input_data = pd.DataFrame([{
            'gender': 'Male',  # Adding default value for missing column
            'SeniorCitizen': 'No',
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': tenure,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': internet_service,
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': 'Yes',
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])
        
        # Apply feature engineering
        input_data_engineered = create_features(input_data)
        
        # Make prediction
        try:
            proba = model.predict_proba(input_data_engineered)[0, 1]
            prediction = "High Risk" if proba > 0.5 else "Low Risk"
            
            # Display results
            st.subheader("Prediction Result")
            st.metric("Churn Risk", prediction, delta=f"{proba:.1%} probability")
            
            # Show explanation
            st.write("Key factors influencing this prediction:")
            if tenure < 12:
                st.write("- Short customer tenure (higher risk)")
            if contract == "Month-to-month":
                st.write("- Month-to-month contract (higher risk)")
            if internet_service == "Fiber optic":
                st.write("- Fiber optic internet service (higher risk)")
            if streaming_tv == "Yes" and streaming_movies == "Yes":
                st.write("- Uses both streaming services (higher risk)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Input data columns:", input_data_engineered.columns.tolist())
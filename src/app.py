from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

from data.feature_engineering import create_features

# Load the trained model
model = joblib.load('models/best_model.pkl')

# Define input data model
class CustomerData(BaseModel):
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.post("/predict")
async def predict(data: CustomerData):
    # Convert input data to DataFrame
    input_data = data.model_dump()
    df = pd.DataFrame([input_data])
    
    # Apply feature engineering (same as during training)
    df = create_features(df)
    
    # Make prediction
    proba = model.predict_proba(df)[0, 1]
    prediction = int(proba > 0.5)  # Using 0.5 as threshold
    
    return {
        "prediction": prediction,
        "probability": float(proba),
        "churn_risk": "High" if prediction == 1 else "Low"
    }
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API"}

# Helper function for feature engineering (same as during training)
def create_features(df):
    df = df.copy()
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, np.inf], 
                              labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])
    df['charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['streaming_both'] = ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(int)
    return df
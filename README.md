# Customer Churn Prediction Project

## Project Overview
End-to-end data science project predicting customer churn for a telecom company.

## Project Structure
customer_churn_prediction/
├── data/ # Data files
│ ├── raw/ # Raw data (ignored by git)
│ └── processed/ # Processed data (ignored by git)
├── models/ # Trained models (ignored by git)
├── monitoring/ # Monitoring logs and reports
├── notebooks/ # Jupyter notebooks for exploration
├── reports/ # Generated analysis reports
│ └── figures/ # Visualization images
├── src/ # Source code
│ ├── app.py # FastAPI application
│ ├── dashboard.py # Streamlit dashboard
│ ├── monitoring.py # Model monitoring
│ └── run_monitoring.py # Monitoring script
├── .gitignore
├── README.md
└── requirements.txt

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Usage
Run the FastAPI server:

uvicorn src.app:app --reload

Run the Streamlit dashboard:
streamlit run src/dashboard.py

Run monitoring:
python src/run_monitoring.py


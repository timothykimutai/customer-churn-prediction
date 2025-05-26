# 📉 Customer Churn Prediction Project

## 🚀 Project Overview

This is an end-to-end data science project focused on predicting customer churn for a telecom company. It includes data cleaning, feature engineering, model training, evaluation, monitoring, and visualization dashboards using FastAPI and Streamlit.

---

## 🗂️ Project Structure

```
customer_churn_prediction/
├── data/                  # Data files
│   ├── raw/               # Raw data (ignored by git)
│   └── processed/         # Processed/cleaned data (ignored by git)
├── models/                # Trained model artifacts (ignored by git)
├── monitoring/            # Monitoring logs and drift reports
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── reports/               # Analysis reports
│   └── figures/           # Plots and visualizations
├── src/                   # Source code
│   ├── app.py             # FastAPI backend for predictions
│   ├── dashboard.py       # Streamlit dashboard
│   ├── monitoring.py      # Model monitoring utilities
│   └── run_monitoring.py  # Script to run monitoring checks
├── .gitignore             # Files/folders to ignore in version control
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/timothykimutai/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚦 Usage

### ▶️ Run the FastAPI server

```bash
uvicorn src.app:app --reload
```

### 📊 Launch the Streamlit dashboard

```bash
streamlit run src/dashboard.py
```

### 📉 Execute the monitoring script

```bash
python src/run_monitoring.py
```


# run_monitoring.py
from monitoring import ModelMonitor
import pandas as pd
from datetime import datetime

def main():
    print(f"Running scheduled model monitoring at {datetime.now()}")
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # In production, you would load fresh data from your database/API
    # For this example, we'll use a sample from our existing data
    new_data = pd.read_csv('data/cleaned_churn_data.csv').sample(frac=0.2)
    X_new = new_data.drop('Churn', axis=1)
    y_new = new_data['Churn']
    
    # Log performance
    metrics = monitor.log_performance(X_new, y_new, f"batch_{datetime.now().strftime('%Y%m%d')}")
    
    if metrics:
        print(f"Logged metrics: F1={metrics['f1_score']:.3f}, AUC={metrics['roc_auc']:.3f}")
    else:
        print("Failed to log metrics")
    
    # Generate report
    report = monitor.generate_performance_report()
    print("\nPerformance Report:")
    print(report)

if __name__ == "__main__":
    main()
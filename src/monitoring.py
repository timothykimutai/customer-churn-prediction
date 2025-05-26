# monitoring.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score
import os
import warnings

class ModelMonitor:
    def __init__(self, model_path='models/best_model.pkl', reference_data_path='data/cleaned_churn_data.csv'):
        self.model = joblib.load(model_path)
        self.reference_data = pd.read_csv(reference_data_path)
        
        # Calculate reference metrics from the validation set
        X_ref = create_features(self.reference_data.drop('Churn', axis=1))
        y_ref = self.reference_data['Churn']
        y_pred_ref = self.model.predict(X_ref)
        y_proba_ref = self.model.predict_proba(X_ref)[:, 1]
        
        self.reference_metrics = {
            'f1_score': f1_score(y_ref, y_pred_ref),
            'roc_auc': roc_auc_score(y_ref, y_proba_ref),
            'sample_size': len(X_ref)
        }
        
        # Create monitoring directory if it doesn't exist
        os.makedirs('monitoring', exist_ok=True)
        
        # Initialize performance log
        self.performance_log_file = 'monitoring/performance_log.csv'
        if os.path.exists(self.performance_log_file):
            self.performance_log = pd.read_csv(self.performance_log_file)
        else:
            self.performance_log = pd.DataFrame(columns=[
                'timestamp', 'dataset', 'f1_score', 'roc_auc', 'sample_size'
            ])
    
    def log_performance(self, X, y_true, dataset_name):
        """Log model performance on new data"""
        try:
            # Ensure we have the engineered features
            X_fe = create_features(X)
            
            # Make predictions
            y_pred = self.model.predict(X_fe)
            y_proba = self.model.predict_proba(X_fe)[:, 1]
            
            current_metrics = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': dataset_name,
                'f1_score': f1_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_proba),
                'sample_size': len(X_fe)
            }
            
            # Append to log
            self.performance_log = pd.concat([
                self.performance_log,
                pd.DataFrame([current_metrics])
            ], ignore_index=True)
            
            # Check for drift
            self.check_for_drift(current_metrics)
            
            # Save updated log
            self.save_performance_log()
            
            return current_metrics
        
        except Exception as e:
            warnings.warn(f"Error in log_performance: {str(e)}")
            return None
    
    def check_for_drift(self, current_metrics):
        """Check for significant performance degradation"""
        try:
            f1_drift = (self.reference_metrics['f1_score'] - current_metrics['f1_score']) / self.reference_metrics['f1_score']
            auc_drift = (self.reference_metrics['roc_auc'] - current_metrics['roc_auc']) / self.reference_metrics['roc_auc']
            
            threshold = 0.15  # 15% degradation
            alerts = []
            
            if f1_drift > threshold:
                alerts.append(f"Significant F1-score degradation: {f1_drift:.1%}")
            if auc_drift > threshold:
                alerts.append(f"Significant ROC-AUC degradation: {auc_drift:.1%}")
                
            if alerts:
                alert_msg = f"ALERT: Model performance drift detected!\n" + "\n".join(alerts)
                print(alert_msg)
                # In production, would trigger alerting system here
                with open('monitoring/alerts.log', 'a') as f:
                    f.write(f"{datetime.now()}: {alert_msg}\n")
        
        except Exception as e:
            warnings.warn(f"Error in check_for_drift: {str(e)}")
    
    def get_performance_history(self):
        """Return performance history as DataFrame"""
        return self.performance_log
    
    def save_performance_log(self, path=None):
        """Save performance log to CSV"""
        path = path or self.performance_log_file
        self.performance_log.to_csv(path, index=False)
    
    def generate_performance_report(self):
        """Generate a summary performance report"""
        if len(self.performance_log) == 0:
            return "No performance data available yet."
        
        report = f"Model Performance Monitoring Report\n"
        report += f"Generated at: {datetime.now()}\n\n"
        report += f"Reference Metrics:\n"
        report += f"- F1-score: {self.reference_metrics['f1_score']:.3f}\n"
        report += f"- ROC-AUC: {self.reference_metrics['roc_auc']:.3f}\n"
        report += f"- Sample size: {self.reference_metrics['sample_size']}\n\n"
        
        report += "Performance History Summary:\n"
        report += self.performance_log.describe().to_string()
        
        # Save report
        with open('monitoring/performance_report.txt', 'w') as f:
            f.write(report)
        
        return report

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

# Simulate running monitoring in production
if __name__ == "__main__":
    print("Running model monitoring...")
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Load some new data (in production, this would come from your data pipeline)
    new_data = pd.read_csv('data/cleaned_churn_data.csv').sample(frac=0.3, random_state=42)
    X_new = new_data.drop('Churn', axis=1)
    y_new = new_data['Churn']
    
    # Log performance on this new data
    print("\nLogging performance on new data...")
    metrics = monitor.log_performance(X_new, y_new, 'new_customers_batch_1')
    print("Current performance:", metrics)
    
    # Generate report
    print("\nGenerating performance report...")
    report = monitor.generate_performance_report()
    print(report)
    
    print("\nMonitoring complete. Data saved to monitoring/ directory")
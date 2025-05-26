import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import results, best_models
from data.feature_engineering import cat_cols, num_cols
import joblib
# Compare model performance
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'F1-Score': [res['f1_score'] for res in results.values()],
    'ROC-AUC': [res['roc_auc'] for res in results.values()]
}).sort_values('F1-Score', ascending=False)

print("\nModel Performance Comparison:")
print(performance_df.to_string(index=False))

# Plot performance comparison
plt.figure(figsize=(10, 5))
performance_df.set_index('Model').plot(kind='bar', rot=45)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('reports/figures/model_performance.png')
plt.show()

# Select best model
best_model_name = performance_df.iloc[0]['Model']
best_model = best_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')

# Feature importance for tree-based models
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # Get feature names after one-hot encoding
    ohe_features = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_features = num_cols + list(ohe_features)
    
    # Get feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.show()
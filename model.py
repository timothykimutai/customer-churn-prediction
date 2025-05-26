from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, f1_score, precision_recall_curve)
import joblib
from data.feature_engineering import X_fe, y
from data.feature_engineering import num_cols, cat_cols
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y, test_size=0.2, random_state=42, stratify=y)
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
# Model pipeline function
def build_model_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=sum(y==0)/sum(y==1))
}
# Hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6],
        'classifier__learning_rate': [0.01, 0.1]
    }
}
# Train and tune models
results = {}
best_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = build_model_pipeline(model)
    
    # Grid search with 5-fold CV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grids[name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Store results
    best_models[name] = grid_search.best_estimator_
    y_pred = best_models[name].predict(X_test)
    y_proba = best_models[name].predict_proba(X_test)[:, 1]
    
    results[name] = {
        'best_params': grid_search.best_params_,
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test F1-score: {results[name]['f1_score']:.3f}")
    print(f"Test ROC-AUC: {results[name]['roc_auc']:.3f}")
# Save best models
for name, model in best_models.items():
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.pkl')
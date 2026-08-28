import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_score, recall_score, f1_score
import joblib

def load_and_engineer_features(csv_path="payment_failures.csv"):
    df = pd.read_csv(csv_path)
    
    # 1. Feature Engineering (Datetime)
    # Extract temporal features to allow the model to discover payday/weekend effects
    df['failed_at'] = pd.to_datetime(df['failed_at'])
    df['day_of_month'] = df['failed_at'].dt.day
    df['day_of_week'] = df['failed_at'].dt.dayofweek
    df['hour_of_day'] = df['failed_at'].dt.hour
    
    # 2. Select Safe Features and Target (Strictly excluding leakage)
    features = [
        'amount', 
        'payment_method', 
        'error_code', 
        'customer_historical_success_rate', 
        'retry_attempt_number', 
        'day_of_month', 
        'day_of_week', 
        'hour_of_day',
        'recovery_action_taken'  # The action chosen becomes a feature
    ]
    target = 'recovery_success'
    
    X = df[features]
    y = df[target].astype(int)
    
    return X, y

def build_preprocessor():
    """Creates a scikit-learn preprocessing pipeline for categorical and numerical data."""
    numeric_features = [
        'amount', 'customer_historical_success_rate', 'retry_attempt_number', 
        'day_of_month', 'day_of_week', 'hour_of_day'
    ]
    categorical_features = ['payment_method', 'error_code', 'recovery_action_taken']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            # drop='first' avoids dummy variable trap for linear models
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
    return preprocessor

def evaluate_model(model, X, y, dataset_name=""):
    """Calculates and prints standard classification metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = {
        'ROC-AUC': roc_auc_score(y, y_prob),
        'PR-AUC': average_precision_score(y, y_prob),
        'Log Loss': log_loss(y, y_prob),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0)
    }
    
    print(f"\n--- {dataset_name} Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return metrics

def run_training_pipeline():
    print("Loading data and engineering features...")
    X, y = load_and_engineer_features()
    
    # 3. Reproducible Splitting: 70% Train, 15% Validation, 15% Test
    # The test set is held entirely apart.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Split the remaining 85% into Train (70%) and Val (15%) -> 0.15 / 0.85 = ~0.1765
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)
    
    preprocessor = build_preprocessor()
    
    # 4. Train Simple Baseline Model (Logistic Regression)
    print("\nTraining Simple Baseline (Logistic Regression)...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_val_metrics = evaluate_model(lr_pipeline, X_val, y_val, "Logistic Regression (Validation Set)")
    
    # 5. Train XGBoost Model
    print("\nTraining XGBoost Model...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            eval_metric='logloss', 
            random_state=42, 
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_val_metrics = evaluate_model(xgb_pipeline, X_val, y_val, "XGBoost (Validation Set)")
    
    # 6. Model Selection (Based purely on Validation Set)
    print("\n================ MODEL SELECTION ================")
    if xgb_val_metrics['ROC-AUC'] > lr_val_metrics['ROC-AUC']:
        print("Winner: XGBoost selected based on superior Validation ROC-AUC and non-linear capability.")
        best_model = xgb_pipeline
    else:
        print("Winner: Logistic Regression selected.")
        best_model = lr_pipeline
        
    # 7. Final Evaluation on Held-out Test Set
    print("\n================ FINAL TEST EVALUATION ================")
    print("Evaluating the selected winner on the strictly held-out Test Set...")
    test_metrics = evaluate_model(best_model, X_test, y_test, "Winning Model (Test Set)")
    
    # 8. Save the Pipeline
    model_path = "recovery_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n ML Training complete! Complete preprocessing and model pipeline saved to '{model_path}'")

if __name__ == '__main__':
    run_training_pipeline()

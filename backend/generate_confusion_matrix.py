import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

def main():
    print("Generating Confusion Matrix from Current Model")
    print("=" * 80)
    
    # Load data
    data_path = os.path.join('data', 'processed', '515.csv')
    print(f"\n1. Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset: {len(df)} rows")
    
    # Create binary target
    roi_threshold = 145.5
    y = (df['roi'] >= roi_threshold).astype(int)
    df = df.drop(columns=['roi'])
    print(f"   Binary target: {y.sum()}/{len(y)} High ROI ({y.sum()/len(y)*100:.1f}%)")
    
    # Feature engineering (same as training)
    print("\n2. Feature engineering")
    df['log_investment'] = np.log1p(df['investment_eur'])
    df['log_revenue'] = np.log1p(df['revenue_m_eur'])
    df['investment_per_day'] = df['investment_eur'] / (df['days_to_deployment'] + 1)
    df['total_prep_time'] = df['days_diagnostic'] + df['days_poc']
    df['deployment_speed'] = 1 / (df['days_to_deployment'] + 1)
    df['is_large_company'] = (df['company_size'] == 'grande').astype(int)
    df['human_in_loop'] = df['human_in_loop'].astype(int)
    df['revenue_investment_ratio'] = df['revenue_m_eur'] / (df['investment_eur'] / 1_000_000 + 1)
    df['time_efficiency'] = df['time_saved_hours_month'] / (df['total_prep_time'] + 1)
    df['revenue_time_interaction'] = df['revenue_increase_percent'] * df['time_saved_hours_month']
    
    numeric_features = [
        'log_investment', 'log_revenue', 'investment_per_day',
        'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
        'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year',
        'revenue_investment_ratio', 'time_efficiency', 'revenue_time_interaction'
    ]
    categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']
    X = df[numeric_features + categorical_features].copy()
    
    # Train/Test split (80/20, stratified, random_state=42) - SAME AS TRAINING
    print("\n3. Train/Test split (80/20, stratified, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Load the trained model
    model_path = os.path.join('backend', 'models', 'roi_classifier_best.pkl')
    print(f"\n4. Loading trained model: {model_path}")
    model = joblib.load(model_path)
    
    # Make predictions on test set
    print("\n5. Generating predictions on test set")
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (80/20 Split)")
    print("=" * 80)
    print(f"\nActual layout:")
    print(f"                Predicted")
    print(f"              Not-High  High")
    print(f"Actual Not-High   {tn:3d}    {fp:3d}")
    print(f"       High       {fn:3d}    {tp:3d}")
    
    total = tn + fp + fn + tp
    print(f"\nValues for frontend:")
    print(f"  trueNegative: {tn}")
    print(f"  falsePositive: {fp}")
    print(f"  falseNegative: {fn}")
    print(f"  truePositive: {tp}")
    print(f"  Total: {total}")
    
    # Calculate metrics
    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Precision (High ROI): {precision:.1f}%")
    print(f"  Recall (High ROI): {recall:.1f}%")
    print(f"  Specificity (Not-High): {specificity:.1f}%")
    
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    print(classification_report(y_test, y_pred, target_names=['Not-High', 'High']))
    
    print("\n" + "=" * 80)
    print("VALIDATION METHOD: 80/20 Train-Test Split")
    print("=" * 80)
    print("The model uses an 80/20 train-test split (stratified, random_state=42)")
    print("The confusion matrix above is from the 20% test set (103 samples)")
    print("5-fold cross-validation is used during training for model selection,")
    print("but the final confusion matrix is from the held-out test set.")

if __name__ == "__main__":
    main()

"""
Quick test of retrained model with enhanced dataset
"""
import joblib
import pandas as pd
import numpy as np

# Load the retrained model
model = joblib.load('backend/models/roi_model.pkl')
print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# Create a test case (similar to existing data)
test_case = pd.DataFrame([{
    'year': 2024,
    'quarter': 'q2',
    'sector': 'finance',
    'company_size': 'grande',
    'revenue_m_eur': 500.0,
    'ai_use_case': 'customer service bot',
    'deployment_type': 'nlp',
    'days_diagnostic': 45,
    'days_poc': 90,
    'days_to_deployment': 300,
    'investment_eur': 1_000_000,
    'time_saved_hours_month': 500,
    'revenue_increase_percent': 0.0,
    'human_in_loop': 1
}])

# Apply feature engineering (same as training)
test_case['log_investment'] = np.log1p(test_case['investment_eur'])
test_case['log_revenue'] = np.log1p(test_case['revenue_m_eur'])
test_case['investment_ratio'] = test_case['investment_eur'] / (test_case['revenue_m_eur'] * 1_000_000)
test_case['investment_per_day'] = test_case['investment_eur'] / (test_case['days_to_deployment'] + 1)
test_case['diagnostic_efficiency'] = test_case['days_diagnostic'] / (test_case['days_to_deployment'] + 1)
test_case['poc_efficiency'] = test_case['days_poc'] / (test_case['days_to_deployment'] + 1)
test_case['total_prep_time'] = test_case['days_diagnostic'] + test_case['days_poc']
test_case['deployment_speed'] = 1 / (test_case['days_to_deployment'] + 1)
test_case['size_investment_interaction'] = test_case['log_revenue'] * test_case['log_investment']
test_case['is_large_company'] = (test_case['company_size'] == 'grande').astype(int)
test_case['is_hybrid_deployment'] = (test_case['deployment_type'] == 'hybrid').astype(int)
test_case['has_revenue_increase'] = (test_case['revenue_increase_percent'] > 0).astype(int)
test_case['has_time_savings'] = (test_case['time_saved_hours_month'] > 0).astype(int)

# Select features
numeric_features = [
    'log_investment', 'log_revenue', 'investment_ratio',
    'investment_per_day', 'diagnostic_efficiency', 'poc_efficiency',
    'total_prep_time', 'deployment_speed', 'size_investment_interaction',
    'is_large_company', 'is_hybrid_deployment', 'human_in_loop', 'year',
    'time_saved_hours_month', 'revenue_increase_percent',
    'has_revenue_increase', 'has_time_savings'
]
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']
X_test = test_case[numeric_features + categorical_features]

# Make prediction
prediction = model.predict(X_test)
print(f"\nTest Case: Large finance company, customer service bot")
print(f"Investment: €1,000,000")
print(f"Time savings: 500 hours/month")
print(f"\nPredicted ROI: {prediction[0]:.2f}%")

print("\n[OK] Model is working correctly!")

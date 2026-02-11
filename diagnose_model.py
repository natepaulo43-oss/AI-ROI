import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("=" * 80)
print("MODEL PERFORMANCE DIAGNOSTIC")
print("=" * 80)

# Load data
df = pd.read_csv('data/processed/ai_roi_modeling_dataset.csv')
print(f"\n1. Dataset shape: {df.shape}")

# Target analysis
print(f"\n2. Target variable (roi) analysis:")
print(df['roi'].describe())
print(f"   Negative ROI count: {(df['roi'] < 0).sum()} ({(df['roi'] < 0).sum()/len(df)*100:.1f}%)")
print(f"   Zero ROI count: {(df['roi'] == 0).sum()}")

# Feature correlations
print(f"\n3. Feature correlations with ROI:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('roi')
for col in numeric_cols:
    corr = df[col].corr(df['roi'])
    print(f"   {col:30s}: {corr:7.4f}")

# Check feature engineering
print(f"\n4. Testing feature engineering:")
df_test = df.copy()
df_test['log_investment'] = np.log1p(df_test['investment_eur'])
df_test['log_revenue'] = np.log1p(df_test['revenue_m_eur'])
df_test['investment_ratio'] = df_test['investment_eur'] / df_test['revenue_m_eur']

print(f"   log_investment correlation: {df_test['log_investment'].corr(df_test['roi']):.4f}")
print(f"   log_revenue correlation: {df_test['log_revenue'].corr(df_test['roi']):.4f}")
print(f"   investment_ratio correlation: {df_test['investment_ratio'].corr(df_test['roi']):.4f}")

# Test different model configurations
print(f"\n5. Testing model configurations:")

y = df['roi'].copy()
df_features = df.drop(columns=['roi'])

# Add engineered features
df_features['log_investment'] = np.log1p(df_features['investment_eur'])
df_features['log_revenue'] = np.log1p(df_features['revenue_m_eur'])
df_features['investment_ratio'] = df_features['investment_eur'] / df_features['revenue_m_eur']
df_features['human_in_loop'] = df_features['human_in_loop'].astype(int)

# Configuration 1: Current features
numeric_features = ['log_investment', 'log_revenue', 'investment_ratio', 'human_in_loop',
                   'days_to_deployment', 'days_diagnostic', 'days_poc']
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type']

X = df_features[numeric_features + categorical_features].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Test 1: Current config (max_depth=6)
model1 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1))
])
model1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)
print(f"   Config 1 (max_depth=6): R² = {score1:.4f}")

# Test 2: Deeper trees
model2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=400, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1))
])
model2.fit(X_train, y_train)
score2 = model2.score(X_test, y_test)
print(f"   Config 2 (max_depth=15): R² = {score2:.4f}")

# Test 3: No depth limit
model3 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1))
])
model3.fit(X_train, y_train)
score3 = model3.score(X_test, y_test)
print(f"   Config 3 (max_depth=None): R² = {score3:.4f}")

# Test 4: Add more features (including year, quarter)
numeric_features_extended = numeric_features + ['year']
X_extended = df_features[numeric_features_extended + categorical_features + ['quarter']].copy()
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_extended, y, test_size=0.2, random_state=42)

preprocessor_ext = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features_extended),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features + ['quarter'])
    ])

model4 = Pipeline([
    ('preprocessor', preprocessor_ext),
    ('regressor', RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1))
])
model4.fit(X_train_ext, y_train_ext)
score4 = model4.score(X_test_ext, y_test_ext)
print(f"   Config 4 (with year, quarter): R² = {score4:.4f}")

# Feature importance from best model
print(f"\n6. Feature importance (from best model):")
best_model = model4 if score4 > max(score1, score2, score3) else model3
feature_names = numeric_features_extended + list(best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features + ['quarter']))
importances = best_model.named_steps['regressor'].feature_importances_
feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

for feat, imp in feature_importance[:15]:
    print(f"   {feat:40s}: {imp:.4f}")

print(f"\n7. RECOMMENDATION:")
if max(score1, score2, score3, score4) < 0.3:
    print("   ⚠️  CRITICAL: Model has very poor predictive power (<30% R²)")
    print("   Possible causes:")
    print("   - ROI is highly stochastic and not predictable from pre-adoption features")
    print("   - Need more informative features (e.g., industry benchmarks, team experience)")
    print("   - Dataset too small (200 samples) for complex patterns")
    print("   - Consider simpler baseline models or feature engineering")
else:
    print(f"   ✅ Best configuration achieves R² = {max(score1, score2, score3, score4):.4f}")
    print(f"   Use: max_depth={'None' if score3 > score2 else '15'}, include year/quarter: {score4 > max(score1, score2, score3)}")

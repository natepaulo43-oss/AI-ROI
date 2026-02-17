import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

print("=" * 80)
print("ENHANCED ROI MODEL TRAINING - NEW DATASET")
print("=" * 80)

df = pd.read_csv('data/processed/ai_roi_full_combined_cleaned.csv')
print(f"\n1. Dataset loaded: {df.shape}")
print(f"   ROI range: [{df['roi'].min():.1f}, {df['roi'].max():.1f}]")
print(f"   ROI mean: {df['roi'].mean():.1f}, median: {df['roi'].median():.1f}")

print("\n2. Handling outliers and extreme values")
roi_q1 = df['roi'].quantile(0.05)
roi_q99 = df['roi'].quantile(0.95)
print(f"   5th percentile: {roi_q1:.1f}, 95th percentile: {roi_q99:.1f}")

df_filtered = df[(df['roi'] >= roi_q1) & (df['roi'] <= roi_q99)].copy()
print(f"   Filtered dataset: {df_filtered.shape} (removed {len(df) - len(df_filtered)} outliers)")

y = df_filtered['roi'].copy()
df_work = df_filtered.drop(columns=['roi'])

print("\n3. Feature engineering")
df_work['log_investment'] = np.log1p(df_work['investment_eur'])
df_work['log_revenue'] = np.log1p(df_work['revenue_m_eur'])
df_work['investment_ratio'] = df_work['investment_eur'] / (df_work['revenue_m_eur'] * 1_000_000 + 1)
df_work['investment_per_day'] = df_work['investment_eur'] / (df_work['days_to_deployment'] + 1)
df_work['total_prep_time'] = df_work['days_diagnostic'] + df_work['days_poc']
df_work['deployment_speed'] = 1 / (df_work['days_to_deployment'] + 1)
df_work['is_large_company'] = (df_work['company_size'] == 'grande').astype(int)
df_work['human_in_loop'] = df_work['human_in_loop'].astype(int)

numeric_features = [
    'log_investment', 'log_revenue', 'investment_ratio', 'investment_per_day',
    'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
    'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year'
]
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']

X = df_work[numeric_features + categorical_features].copy()

print(f"   Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

print("\n4. Train/test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

print("\n5. Training multiple models")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=10,
        random_state=42,
        verbose=-1
    )
}

results = {}
best_model = None
best_r2 = -np.inf

for name, regressor in models.items():
    print(f"\n   Training {name}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    
    results[name] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': pipeline
    }
    
    print(f"      R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    print(f"      CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = name

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

for name, res in results.items():
    marker = " ★ BEST" if name == best_model else ""
    print(f"\n{name}{marker}")
    print(f"  Test R²:        {res['r2']:.4f}")
    print(f"  Test MAE:       {res['mae']:.2f}%")
    print(f"  Test RMSE:      {res['rmse']:.2f}%")
    print(f"  CV R² (5-fold): {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model}")
print("=" * 80)

model_dir = 'backend/models'
os.makedirs(model_dir, exist_ok=True)

best_pipeline = results[best_model]['pipeline']
model_path = os.path.join(model_dir, 'roi_model.pkl')
joblib.dump(best_pipeline, model_path)
print(f"\n✓ Best model saved: {model_path}")

for name, res in results.items():
    if name != best_model:
        alt_path = os.path.join(model_dir, f'roi_model_{name.lower()}.pkl')
        joblib.dump(res['pipeline'], alt_path)
        print(f"✓ Alternative model saved: {alt_path}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if best_r2 > 0.5:
    print("✓ GOOD: Model explains >50% of ROI variance")
    print("  → Reliable for production predictions")
elif best_r2 > 0.3:
    print("⚠ MODERATE: Model explains 30-50% of ROI variance")
    print("  → Useful for estimates, but expect significant uncertainty")
elif best_r2 > 0.1:
    print("⚠ WEAK: Model explains 10-30% of ROI variance")
    print("  → Limited predictive power, use with caution")
else:
    print("✗ POOR: Model explains <10% of ROI variance")
    print("  → ROI is highly unpredictable from available features")
    print("  → Consider classification (high/medium/low ROI) instead")

print(f"\nDataset size: {len(df_filtered)} samples (after outlier removal)")
print(f"Features used: {len(numeric_features) + len(categorical_features)}")
print("\nRECOMMENDATION:")
if best_r2 < 0.3:
    print("  • Collect more data to improve model performance")
    print("  • Consider ROI classification instead of regression")
    print("  • Use model for directional guidance, not precise predictions")
else:
    print(f"  • Use {best_model} model for production")
    print("  • Monitor predictions and retrain with new data regularly")

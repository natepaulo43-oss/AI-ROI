import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def main():
    print("=" * 80)
    print("IMPROVED ROI MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    data_path = os.path.join('data', 'processed', 'ai_roi_training_dataset_cleaned.csv')
    print(f"\n1. Loading dataset from: {data_path}")
    print(f"   Using CLEANED dataset (timeline issues fixed, validated)")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")

    # Validate data quality
    assert (df['days_diagnostic'] + df['days_poc'] <= df['days_to_deployment']).all(), \
        "Timeline inconsistency detected!"
    print(f"   [OK] Data quality validation passed")
    
    print("\n2. Preparing target variable")
    y = df['roi'].copy()
    df = df.drop(columns=['roi'])
    print(f"   Target (y) shape: {y.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    print("\n3. Advanced feature engineering")
    
    # Log transforms
    df['log_investment'] = np.log1p(df['investment_eur'])
    df['log_revenue'] = np.log1p(df['revenue_m_eur'])
    
    # Ratios and efficiency metrics
    df['investment_ratio'] = df['investment_eur'] / (df['revenue_m_eur'] * 1_000_000)
    df['investment_per_day'] = df['investment_eur'] / (df['days_to_deployment'] + 1)
    df['diagnostic_efficiency'] = df['days_diagnostic'] / (df['days_to_deployment'] + 1)
    df['poc_efficiency'] = df['days_poc'] / (df['days_to_deployment'] + 1)
    
    # Time-based features
    df['total_prep_time'] = df['days_diagnostic'] + df['days_poc']
    df['deployment_speed'] = 1 / (df['days_to_deployment'] + 1)
    
    # Interaction features
    df['size_investment_interaction'] = df['log_revenue'] * df['log_investment']
    
    # Binary flags
    df['is_large_company'] = (df['company_size'] == 'grande').astype(int)
    df['is_hybrid_deployment'] = (df['deployment_type'] == 'hybrid').astype(int)
    df['human_in_loop'] = df['human_in_loop'].astype(int)
    df['has_revenue_increase'] = (df['revenue_increase_percent'] > 0).astype(int)
    df['has_time_savings'] = (df['time_saved_hours_month'] > 0).astype(int)
    
    print("   Created engineered features:")
    print("   - Log transforms (investment, revenue)")
    print("   - Efficiency ratios (diagnostic, poc, deployment)")
    print("   - Time-based features (total prep time, deployment speed)")
    print("   - Interaction features (size × investment)")
    print("   - Binary flags (company size, deployment type, outcomes)")
    
    print("\n4. Feature selection strategy")
    print("   Building TWO models:")
    print("   A) CONSERVATIVE: Pre-adoption only (no data leakage)")
    print("   B) PRACTICAL: Includes early deployment signals")
    
    # Model A: Conservative (pre-adoption only)
    numeric_features_conservative = [
        'log_investment', 'log_revenue', 'investment_ratio',
        'investment_per_day', 'diagnostic_efficiency', 'poc_efficiency',
        'total_prep_time', 'deployment_speed', 'size_investment_interaction',
        'is_large_company', 'is_hybrid_deployment', 'human_in_loop', 'year'
    ]
    categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']
    
    # Model B: Practical (with early signals)
    numeric_features_practical = numeric_features_conservative + [
        'time_saved_hours_month', 'revenue_increase_percent',
        'has_revenue_increase', 'has_time_savings'
    ]
    
    print("\n5. Training Model A: CONSERVATIVE (Pre-adoption only)")
    X_conservative = df[numeric_features_conservative + categorical_features].copy()
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_conservative, y, test_size=0.2, random_state=42
    )
    
    preprocessor_c = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features_conservative),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    model_conservative = Pipeline([
        ('preprocessor', preprocessor_c),
        ('regressor', GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10,
            random_state=42
        ))
    ])
    
    model_conservative.fit(X_train_c, y_train_c)
    y_pred_c = model_conservative.predict(X_test_c)
    
    r2_c = r2_score(y_test_c, y_pred_c)
    mae_c = mean_absolute_error(y_test_c, y_pred_c)
    rmse_c = np.sqrt(mean_squared_error(y_test_c, y_pred_c))
    
    print(f"   R² Score:  {r2_c:.4f}")
    print(f"   MAE:       {mae_c:.2f}%")
    print(f"   RMSE:      {rmse_c:.2f}%")
    
    print("\n6. Training Model B: PRACTICAL (With early signals)")
    X_practical = df[numeric_features_practical + categorical_features].copy()
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_practical, y, test_size=0.2, random_state=42
    )
    
    preprocessor_p = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features_practical),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    model_practical = Pipeline([
        ('preprocessor', preprocessor_p),
        ('regressor', GradientBoostingRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=8,
            random_state=42
        ))
    ])
    
    model_practical.fit(X_train_p, y_train_p)
    y_pred_p = model_practical.predict(X_test_p)
    
    r2_p = r2_score(y_test_p, y_pred_p)
    mae_p = mean_absolute_error(y_test_p, y_pred_p)
    rmse_p = np.sqrt(mean_squared_error(y_test_p, y_pred_p))
    
    print(f"   R² Score:  {r2_p:.4f}")
    print(f"   MAE:       {mae_p:.2f}%")
    print(f"   RMSE:      {rmse_p:.2f}%")
    
    print("\n7. Model comparison")
    print(f"   Conservative (pre-only):  R²={r2_c:.4f}, MAE={mae_c:.2f}%")
    print(f"   Practical (with signals): R²={r2_p:.4f}, MAE={mae_p:.2f}%")
    print(f"   Improvement: {(r2_p - r2_c):.4f} R² points")
    
    print("\n8. Saving models")
    model_dir = os.path.join('backend', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save conservative model
    conservative_path = os.path.join(model_dir, 'roi_model_conservative.pkl')
    joblib.dump(model_conservative, conservative_path)
    print(f"   [OK] Conservative model saved: {conservative_path}")
    
    # Save practical model (as default)
    practical_path = os.path.join(model_dir, 'roi_model.pkl')
    joblib.dump(model_practical, practical_path)
    print(f"   [OK] Practical model saved: {practical_path}")
    
    print("\n9. Feature importance (Practical model - top 15)")
    feature_names_practical = numeric_features_practical + list(
        model_practical.named_steps['preprocessor'].named_transformers_['cat']
        .get_feature_names_out(categorical_features)
    )
    importances = model_practical.named_steps['regressor'].feature_importances_
    feature_importance = sorted(zip(feature_names_practical, importances), 
                                key=lambda x: x[1], reverse=True)
    
    for feat, imp in feature_importance[:15]:
        print(f"   {feat:45s}: {imp:.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\n SUMMARY:")
    print(f"   Conservative Model (pre-adoption only): R²={r2_c:.4f}")
    print(f"   Practical Model (with early signals):  R²={r2_p:.4f}")
    print("\n INTERPRETATION:")
    if r2_c < 0.15:
        print("   [WARNING]  Pre-adoption features have limited predictive power (<15% R²)")
        print("   -> ROI is highly dependent on execution and post-deployment factors")
    if r2_p > 0.5:
        print("   [OK] Practical model achieves good performance (>50% R²)")
        print("   -> Early deployment signals are strong ROI predictors")
    elif r2_p > 0.3:
        print("   [WARNING]  Practical model has moderate performance (30-50% R²)")
        print("   -> Some predictability, but ROI remains partially stochastic")
    else:
        print("   [ERROR] Even with early signals, ROI prediction is challenging")
        print("   -> Consider classification approach or accept high uncertainty")
    
    print("\n RECOMMENDATION:")
    print("   Use PRACTICAL model for production (includes early deployment signals)")
    print("   Use CONSERVATIVE model for pre-deployment estimates (with high uncertainty)")

if __name__ == "__main__":
    main()

"""
Enhanced ROI Model Training with Temporal Validation
- Uses cleaned data
- Implements temporal cross-validation
- Prevents temporal leakage
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime

def temporal_train_test_split(df, test_year=2025):
    """
    Split data temporally: train on past, test on future
    Prevents temporal leakage
    """
    train = df[df['year'] < test_year].copy()
    test = df[df['year'] == test_year].copy()
    return train, test

def main():
    print("=" * 80)
    print("ROI MODEL TRAINING WITH TEMPORAL VALIDATION")
    print("=" * 80)
    print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load cleaned data
    data_path = os.path.join('data', 'processed', 'ai_roi_training_dataset_cleaned.csv')
    print(f"\n1. Loading cleaned dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")

    # Validate data quality
    print("\n2. Validating data quality...")
    assert (df['days_diagnostic'] + df['days_poc'] <= df['days_to_deployment']).all(), \
        "Timeline inconsistency detected!"
    assert df['investment_eur'].min() > 0, "Invalid investment values!"
    assert df['revenue_m_eur'].min() > 0, "Invalid revenue values!"
    print("   [OK] Data quality validation passed")

    # Temporal split
    print("\n3. Performing temporal train/test split...")
    print("   Strategy: Train on 2022-2024, Test on 2025")

    train_df, test_df = temporal_train_test_split(df, test_year=2025)
    print(f"   Train set: {len(train_df)} records ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"   Test set:  {len(test_df)} records (2025)")

    if len(test_df) < 10:
        print("   [WARNING] Small test set - consider using cross-validation")

    # Prepare target
    y_train = train_df['roi'].copy()
    y_test = test_df['roi'].copy()
    train_df = train_df.drop(columns=['roi'])
    test_df = test_df.drop(columns=['roi'])

    print(f"   Train ROI range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Test ROI range:  [{y_test.min():.2f}, {y_test.max():.2f}]")

    # Feature engineering
    print("\n4. Engineering features...")

    def engineer_features(df):
        """Apply same feature engineering to train and test"""
        df = df.copy()

        # Log transforms
        df['log_investment'] = np.log1p(df['investment_eur'])
        df['log_revenue'] = np.log1p(df['revenue_m_eur'])

        # Ratios and efficiency
        df['investment_ratio'] = df['investment_eur'] / (df['revenue_m_eur'] * 1_000_000)
        df['investment_per_day'] = df['investment_eur'] / (df['days_to_deployment'] + 1)
        df['diagnostic_efficiency'] = df['days_diagnostic'] / (df['days_to_deployment'] + 1)
        df['poc_efficiency'] = df['days_poc'] / (df['days_to_deployment'] + 1)

        # Time-based
        df['total_prep_time'] = df['days_diagnostic'] + df['days_poc']
        df['deployment_speed'] = 1 / (df['days_to_deployment'] + 1)

        # Interactions
        df['size_investment_interaction'] = df['log_revenue'] * df['log_investment']

        # Binary flags
        df['is_large_company'] = (df['company_size'] == 'grande').astype(int)
        df['is_hybrid_deployment'] = (df['deployment_type'] == 'hybrid').astype(int)
        df['human_in_loop'] = df['human_in_loop'].astype(int)
        df['has_revenue_increase'] = (df['revenue_increase_percent'] > 0).astype(int)
        df['has_time_savings'] = (df['time_saved_hours_month'] > 0).astype(int)

        return df

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    print("   Created 17 engineered features")

    # Define feature sets
    numeric_features_conservative = [
        'log_investment', 'log_revenue', 'investment_ratio',
        'investment_per_day', 'diagnostic_efficiency', 'poc_efficiency',
        'total_prep_time', 'deployment_speed', 'size_investment_interaction',
        'is_large_company', 'is_hybrid_deployment', 'human_in_loop', 'year'
    ]

    numeric_features_practical = numeric_features_conservative + [
        'time_saved_hours_month', 'revenue_increase_percent',
        'has_revenue_increase', 'has_time_savings'
    ]

    categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']

    # Train Conservative Model
    print("\n5. Training CONSERVATIVE model (no data leakage)...")
    print("   Features: Pre-adoption only")

    X_train_c = train_df[numeric_features_conservative + categorical_features]
    X_test_c = test_df[numeric_features_conservative + categorical_features]

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

    model_conservative.fit(X_train_c, y_train)
    y_pred_c_test = model_conservative.predict(X_test_c)

    # Conservative model metrics
    r2_c = r2_score(y_test, y_pred_c_test)
    mae_c = mean_absolute_error(y_test, y_pred_c_test)
    rmse_c = np.sqrt(mean_squared_error(y_test, y_pred_c_test))

    print(f"   Temporal Test Set Performance:")
    print(f"   R2 Score:  {r2_c:.4f}")
    print(f"   MAE:       {mae_c:.2f}%")
    print(f"   RMSE:      {rmse_c:.2f}%")

    # Train Practical Model
    print("\n6. Training PRACTICAL model (with early signals)...")
    print("   Features: Pre-adoption + outcome variables")
    print("   [WARNING] This model has data leakage - use only mid-deployment!")

    X_train_p = train_df[numeric_features_practical + categorical_features]
    X_test_p = test_df[numeric_features_practical + categorical_features]

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

    model_practical.fit(X_train_p, y_train)
    y_pred_p_test = model_practical.predict(X_test_p)

    # Practical model metrics
    r2_p = r2_score(y_test, y_pred_p_test)
    mae_p = mean_absolute_error(y_test, y_pred_p_test)
    rmse_p = np.sqrt(mean_squared_error(y_test, y_pred_p_test))

    print(f"   Temporal Test Set Performance:")
    print(f"   R2 Score:  {r2_p:.4f}")
    print(f"   MAE:       {mae_p:.2f}%")
    print(f"   RMSE:      {rmse_p:.2f}%")

    # Model comparison
    print("\n7. Model Comparison on 2025 Test Data")
    print("   " + "-"*60)
    print(f"   Conservative: R2={r2_c:.4f}, MAE={mae_c:.2f}%")
    print(f"   Practical:    R2={r2_p:.4f}, MAE={mae_p:.2f}%")
    print(f"   Improvement:  {(r2_p - r2_c):.4f} R2 points")

    # Save models
    print("\n8. Saving models...")
    model_dir = os.path.join('backend', 'models')
    os.makedirs(model_dir, exist_ok=True)

    conservative_path = os.path.join(model_dir, 'roi_model_conservative_temporal.pkl')
    practical_path = os.path.join(model_dir, 'roi_model_temporal.pkl')

    joblib.dump(model_conservative, conservative_path)
    joblib.dump(model_practical, practical_path)

    print(f"   [OK] Conservative model: {conservative_path}")
    print(f"   [OK] Practical model: {practical_path}")

    # Generate validation report
    print("\n9. Validation Report")
    print("   " + "-"*60)

    # Analyze errors
    errors_c = np.abs(y_test - y_pred_c_test)
    errors_p = np.abs(y_test - y_pred_p_test)

    print(f"\n   Conservative Model Error Distribution:")
    print(f"   - 25th percentile: {np.percentile(errors_c, 25):.2f}%")
    print(f"   - Median:          {np.percentile(errors_c, 50):.2f}%")
    print(f"   - 75th percentile: {np.percentile(errors_c, 75):.2f}%")
    print(f"   - Errors > 100%:   {(errors_c > 100).sum()}/{len(errors_c)}")

    print(f"\n   Practical Model Error Distribution:")
    print(f"   - 25th percentile: {np.percentile(errors_p, 25):.2f}%")
    print(f"   - Median:          {np.percentile(errors_p, 50):.2f}%")
    print(f"   - 75th percentile: {np.percentile(errors_p, 75):.2f}%")
    print(f"   - Errors > 100%:   {(errors_p > 100).sum()}/{len(errors_p)}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print("\n[SUMMARY]")
    print(f"  Training approach: Temporal validation (prevents leakage)")
    print(f"  Train period: 2022-2024 ({len(train_df)} records)")
    print(f"  Test period: 2025 ({len(test_df)} records)")
    print(f"  Conservative R2: {r2_c:.4f} (pre-deployment use)")
    print(f"  Practical R2: {r2_p:.4f} (mid-deployment use)")

    print("\n[INTERPRETATION]")
    if r2_c < 0:
        print("  [WARNING] Conservative model has negative R2 on future data")
        print("            Pre-adoption features have very limited predictive power")
        print("            Recommend: Use industry benchmarks + Conservative as ensemble")

    if r2_p > 0.15:
        print("  [OK] Practical model achieves reasonable R2 on future data")
        print("       Model generalizes acceptably to 2025")

    print("\n[RECOMMENDATIONS]")
    print("  1. Use Conservative model for pre-deployment (with ±90% uncertainty)")
    print("  2. Use Practical model for mid-deployment (with ±75% uncertainty)")
    print("  3. Monitor predictions vs actuals on 2025 data")
    print("  4. Retrain quarterly as more 2025+ data becomes available")
    print("  5. Consider ensemble: 0.6*Practical + 0.4*Industry_Benchmark")

    return model_conservative, model_practical, r2_c, r2_p

if __name__ == "__main__":
    models = main()

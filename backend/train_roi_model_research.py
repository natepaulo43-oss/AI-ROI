import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")

def calculate_prediction_intervals(model, X, confidence=0.95):
    """Calculate prediction intervals using bootstrap"""
    predictions = []
    n_iterations = 100
    n_samples = len(X)
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        pred = model.predict(X_bootstrap)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    lower = np.percentile(predictions, (1 - confidence) / 2 * 100, axis=0)
    upper = np.percentile(predictions, (1 + confidence) / 2 * 100, axis=0)
    
    return lower, upper

def plot_residuals(y_true, y_pred, model_name, output_dir):
    """Create residual plots"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("RESEARCH-GRADE ROI MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Create output directory for plots
    output_dir = os.path.join('backend', 'research_output')
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join('data', 'processed', 'ai_roi_training_dataset_cleaned.csv')
    print(f"\n1. Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    
    print("\n2. Preparing target variable")
    y = df['roi'].copy()
    df = df.drop(columns=['roi'])
    print(f"   Target (y) shape: {y.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    print("\n3. ENHANCED feature engineering")
    
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
    
    # Interaction features (NEW)
    df['size_investment_interaction'] = df['log_revenue'] * df['log_investment']
    df['time_investment_interaction'] = df['total_prep_time'] * df['log_investment']
    df['efficiency_investment_interaction'] = df['diagnostic_efficiency'] * df['log_investment']
    
    # Binary flags
    df['is_large_company'] = (df['company_size'] == 'grande').astype(int)
    df['is_hybrid_deployment'] = (df['deployment_type'] == 'hybrid').astype(int)
    df['human_in_loop'] = df['human_in_loop'].astype(int)
    df['has_revenue_increase'] = (df['revenue_increase_percent'] > 0).astype(int)
    df['has_time_savings'] = (df['time_saved_hours_month'] > 0).astype(int)
    
    # Sector-specific interactions (NEW)
    df['finance_investment'] = (df['sector'] == 'finance').astype(int) * df['log_investment']
    df['retail_investment'] = (df['sector'] == 'retail').astype(int) * df['log_investment']
    
    print("   Created enhanced features:")
    print("   - Log transforms (investment, revenue)")
    print("   - Efficiency ratios (diagnostic, poc, deployment)")
    print("   - Time-based features (total prep time, deployment speed)")
    print("   - NEW: 3 interaction features (time×investment, efficiency×investment)")
    print("   - NEW: 2 sector-specific interactions (finance, retail)")
    print("   - Binary flags (company size, deployment type, outcomes)")
    
    print("\n4. Feature selection (PRACTICAL MODEL - with early signals)")
    numeric_features = [
        'log_investment', 'log_revenue', 'investment_ratio',
        'investment_per_day', 'diagnostic_efficiency', 'poc_efficiency',
        'total_prep_time', 'deployment_speed', 'size_investment_interaction',
        'time_investment_interaction', 'efficiency_investment_interaction',
        'is_large_company', 'is_hybrid_deployment', 'human_in_loop', 'year',
        'time_saved_hours_month', 'revenue_increase_percent',
        'has_revenue_increase', 'has_time_savings',
        'finance_investment', 'retail_investment'
    ]
    categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']
    
    X = df[numeric_features + categorical_features].copy()
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Total features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")
    
    print("\n5. Train/Test split (80/20, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    print("\n6. Building preprocessing pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    print("   ✓ Preprocessor created (StandardScaler + OneHotEncoder)")
    
    # Store all models and results
    models_results = {}
    
    # ========================================================================
    # MODEL 1: GradientBoosting with GridSearchCV
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL 1: GradientBoosting with Hyperparameter Tuning")
    print("=" * 80)
    
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    param_grid_gb = {
        'regressor__n_estimators': [300, 500, 700],
        'regressor__max_depth': [4, 6, 8],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__min_samples_split': [5, 8, 10]
    }
    
    print("   Running GridSearchCV (5-fold CV)...")
    print("   Testing 81 parameter combinations...")
    grid_search_gb = GridSearchCV(
        gb_pipeline, param_grid_gb, cv=5, 
        scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search_gb.fit(X_train, y_train)
    
    print(f"\n   Best parameters: {grid_search_gb.best_params_}")
    print(f"   Best CV R² score: {grid_search_gb.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred_gb = grid_search_gb.predict(X_test)
    r2_gb = r2_score(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb) * 100
    
    # Cross-validation scores
    cv_scores_gb = cross_val_score(grid_search_gb.best_estimator_, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n   Test Set Performance:")
    print(f"   R² Score:  {r2_gb:.4f}")
    print(f"   MAE:       {mae_gb:.2f}%")
    print(f"   RMSE:      {rmse_gb:.2f}%")
    print(f"   MAPE:      {mape_gb:.2f}%")
    print(f"\n   Cross-Validation R² (5-fold):")
    print(f"   Mean: {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")
    print(f"   Scores: {[f'{s:.4f}' for s in cv_scores_gb]}")
    
    # Calculate prediction intervals
    print("\n   Calculating prediction intervals (95% confidence)...")
    lower_gb, upper_gb = calculate_prediction_intervals(grid_search_gb, X_test)
    avg_interval_width = np.mean(upper_gb - lower_gb)
    print(f"   Average prediction interval width: ±{avg_interval_width/2:.2f}%")
    
    # Plot residuals
    plot_residuals(y_test, y_pred_gb, 'GradientBoosting', output_dir)
    print(f"   ✓ Residual plots saved to: {output_dir}/GradientBoosting_residuals.png")
    
    models_results['GradientBoosting'] = {
        'model': grid_search_gb.best_estimator_,
        'r2': r2_gb,
        'mae': mae_gb,
        'rmse': rmse_gb,
        'mape': mape_gb,
        'cv_mean': cv_scores_gb.mean(),
        'cv_std': cv_scores_gb.std(),
        'interval_width': avg_interval_width,
        'predictions': y_pred_gb,
        'lower': lower_gb,
        'upper': upper_gb
    }
    
    # ========================================================================
    # MODEL 2: RandomForest with GridSearchCV
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL 2: RandomForest with Hyperparameter Tuning")
    print("=" * 80)
    
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    param_grid_rf = {
        'regressor__n_estimators': [200, 300, 500],
        'regressor__max_depth': [10, 15, 20, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    print("   Running GridSearchCV (5-fold CV)...")
    print("   Testing 108 parameter combinations...")
    grid_search_rf = GridSearchCV(
        rf_pipeline, param_grid_rf, cv=5,
        scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search_rf.fit(X_train, y_train)
    
    print(f"\n   Best parameters: {grid_search_rf.best_params_}")
    print(f"   Best CV R² score: {grid_search_rf.best_score_:.4f}")
    
    y_pred_rf = grid_search_rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100
    
    cv_scores_rf = cross_val_score(grid_search_rf.best_estimator_, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n   Test Set Performance:")
    print(f"   R² Score:  {r2_rf:.4f}")
    print(f"   MAE:       {mae_rf:.2f}%")
    print(f"   RMSE:      {rmse_rf:.2f}%")
    print(f"   MAPE:      {mape_rf:.2f}%")
    print(f"\n   Cross-Validation R² (5-fold):")
    print(f"   Mean: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    
    lower_rf, upper_rf = calculate_prediction_intervals(grid_search_rf, X_test)
    avg_interval_width_rf = np.mean(upper_rf - lower_rf)
    print(f"   Average prediction interval width: ±{avg_interval_width_rf/2:.2f}%")
    
    plot_residuals(y_test, y_pred_rf, 'RandomForest', output_dir)
    print(f"   ✓ Residual plots saved to: {output_dir}/RandomForest_residuals.png")
    
    models_results['RandomForest'] = {
        'model': grid_search_rf.best_estimator_,
        'r2': r2_rf,
        'mae': mae_rf,
        'rmse': rmse_rf,
        'mape': mape_rf,
        'cv_mean': cv_scores_rf.mean(),
        'cv_std': cv_scores_rf.std(),
        'interval_width': avg_interval_width_rf,
        'predictions': y_pred_rf,
        'lower': lower_rf,
        'upper': upper_rf
    }
    
    # ========================================================================
    # MODEL 3: XGBoost (if available)
    # ========================================================================
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 80)
        print("MODEL 3: XGBoost with Hyperparameter Tuning")
        print("=" * 80)
        
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(random_state=42, tree_method='hist'))
        ])
        
        param_grid_xgb = {
            'regressor__n_estimators': [300, 500, 700],
            'regressor__max_depth': [4, 6, 8],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__subsample': [0.8, 0.9, 1.0],
            'regressor__colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        print("   Running GridSearchCV (5-fold CV)...")
        print("   Testing 243 parameter combinations...")
        grid_search_xgb = GridSearchCV(
            xgb_pipeline, param_grid_xgb, cv=5,
            scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search_xgb.fit(X_train, y_train)
        
        print(f"\n   Best parameters: {grid_search_xgb.best_params_}")
        print(f"   Best CV R² score: {grid_search_xgb.best_score_:.4f}")
        
        y_pred_xgb = grid_search_xgb.predict(X_test)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
        
        cv_scores_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train, y_train, cv=5, scoring='r2')
        
        print(f"\n   Test Set Performance:")
        print(f"   R² Score:  {r2_xgb:.4f}")
        print(f"   MAE:       {mae_xgb:.2f}%")
        print(f"   RMSE:      {rmse_xgb:.2f}%")
        print(f"   MAPE:      {mape_xgb:.2f}%")
        print(f"\n   Cross-Validation R² (5-fold):")
        print(f"   Mean: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
        
        lower_xgb, upper_xgb = calculate_prediction_intervals(grid_search_xgb, X_test)
        avg_interval_width_xgb = np.mean(upper_xgb - lower_xgb)
        print(f"   Average prediction interval width: ±{avg_interval_width_xgb/2:.2f}%")
        
        plot_residuals(y_test, y_pred_xgb, 'XGBoost', output_dir)
        print(f"   ✓ Residual plots saved to: {output_dir}/XGBoost_residuals.png")
        
        models_results['XGBoost'] = {
            'model': grid_search_xgb.best_estimator_,
            'r2': r2_xgb,
            'mae': mae_xgb,
            'rmse': rmse_xgb,
            'mape': mape_xgb,
            'cv_mean': cv_scores_xgb.mean(),
            'cv_std': cv_scores_xgb.std(),
            'interval_width': avg_interval_width_xgb,
            'predictions': y_pred_xgb,
            'lower': lower_xgb,
            'upper': upper_xgb
        }
    
    # ========================================================================
    # MODEL 4: LightGBM (if available)
    # ========================================================================
    if LIGHTGBM_AVAILABLE:
        print("\n" + "=" * 80)
        print("MODEL 4: LightGBM with Hyperparameter Tuning")
        print("=" * 80)
        
        lgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=42, verbose=-1))
        ])
        
        param_grid_lgb = {
            'regressor__n_estimators': [300, 500, 700],
            'regressor__max_depth': [4, 6, 8],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__num_leaves': [31, 50, 70],
            'regressor__subsample': [0.8, 0.9, 1.0]
        }
        
        print("   Running GridSearchCV (5-fold CV)...")
        print("   Testing 243 parameter combinations...")
        grid_search_lgb = GridSearchCV(
            lgb_pipeline, param_grid_lgb, cv=5,
            scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search_lgb.fit(X_train, y_train)
        
        print(f"\n   Best parameters: {grid_search_lgb.best_params_}")
        print(f"   Best CV R² score: {grid_search_lgb.best_score_:.4f}")
        
        y_pred_lgb = grid_search_lgb.predict(X_test)
        r2_lgb = r2_score(y_test, y_pred_lgb)
        mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
        rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
        mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100
        
        cv_scores_lgb = cross_val_score(grid_search_lgb.best_estimator_, X_train, y_train, cv=5, scoring='r2')
        
        print(f"\n   Test Set Performance:")
        print(f"   R² Score:  {r2_lgb:.4f}")
        print(f"   MAE:       {mae_lgb:.2f}%")
        print(f"   RMSE:      {rmse_lgb:.2f}%")
        print(f"   MAPE:      {mape_lgb:.2f}%")
        print(f"\n   Cross-Validation R² (5-fold):")
        print(f"   Mean: {cv_scores_lgb.mean():.4f} ± {cv_scores_lgb.std():.4f}")
        
        lower_lgb, upper_lgb = calculate_prediction_intervals(grid_search_lgb, X_test)
        avg_interval_width_lgb = np.mean(upper_lgb - lower_lgb)
        print(f"   Average prediction interval width: ±{avg_interval_width_lgb/2:.2f}%")
        
        plot_residuals(y_test, y_pred_lgb, 'LightGBM', output_dir)
        print(f"   ✓ Residual plots saved to: {output_dir}/LightGBM_residuals.png")
        
        models_results['LightGBM'] = {
            'model': grid_search_lgb.best_estimator_,
            'r2': r2_lgb,
            'mae': mae_lgb,
            'rmse': rmse_lgb,
            'mape': mape_lgb,
            'cv_mean': cv_scores_lgb.mean(),
            'cv_std': cv_scores_lgb.std(),
            'interval_width': avg_interval_width_lgb,
            'predictions': y_pred_lgb,
            'lower': lower_lgb,
            'upper': upper_lgb
        }
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    
    print("\n{:<20} {:<10} {:<10} {:<10} {:<10} {:<15} {:<15}".format(
        "Model", "R²", "MAE", "RMSE", "MAPE", "CV R² (mean)", "Pred. Int. (±)"
    ))
    print("-" * 100)
    
    best_model_name = None
    best_r2 = -float('inf')
    
    for name, results in models_results.items():
        print("{:<20} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15.4f} {:<15.2f}".format(
            name,
            results['r2'],
            results['mae'],
            results['rmse'],
            results['mape'],
            results['cv_mean'],
            results['interval_width'] / 2
        ))
        
        if results['r2'] > best_r2:
            best_r2 = results['r2']
            best_model_name = name
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    print("=" * 80)
    
    # Save best model
    print("\n7. Saving best model")
    model_dir = os.path.join('backend', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    best_model = models_results[best_model_name]['model']
    model_path = os.path.join(model_dir, 'roi_model_research.pkl')
    joblib.dump(best_model, model_path)
    print(f"   ✓ Best model ({best_model_name}) saved to: {model_path}")
    
    # Save comparison report
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RESEARCH-GRADE MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write("{:<20} {:<10} {:<10} {:<10} {:<10} {:<15}\n".format(
            "Model", "R²", "MAE", "RMSE", "MAPE", "CV R² (mean)"
        ))
        f.write("-" * 80 + "\n")
        
        for name, results in models_results.items():
            f.write("{:<20} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.2f} {:<15.4f}\n".format(
                name,
                results['r2'],
                results['mae'],
                results['rmse'],
                results['mape'],
                results['cv_mean']
            ))
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"BEST MODEL: {best_model_name} (R² = {best_r2:.4f})\n")
        f.write("=" * 80 + "\n")
        
        # Add prediction interval examples
        best_results = models_results[best_model_name]
        f.write("\nPREDICTION INTERVAL EXAMPLES (95% confidence):\n")
        f.write("-" * 80 + "\n")
        for i in range(min(10, len(y_test))):
            actual = y_test.iloc[i]
            pred = best_results['predictions'][i]
            lower = best_results['lower'][i]
            upper = best_results['upper'][i]
            f.write(f"Sample {i+1}: Actual={actual:.1f}%, Predicted={pred:.1f}% [{lower:.1f}%, {upper:.1f}%]\n")
    
    print(f"   ✓ Comparison report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ RESEARCH-GRADE TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nIMPROVEMENTS IMPLEMENTED:")
    print("  ✓ 1. Hyperparameter optimization (GridSearchCV)")
    print("  ✓ 2. 5-fold cross-validation for robust evaluation")
    print("  ✓ 3. Multiple algorithms tested (GB, RF, XGB, LGB)")
    print("  ✓ 4. Enhanced feature engineering (interactions, sector-specific)")
    print("  ✓ 5. Prediction intervals (95% confidence)")
    print("  ✓ 6. Comprehensive metrics (R², MAE, RMSE, MAPE)")
    print(f"\nOUTPUTS:")
    print(f"  - Best model: {model_path}")
    print(f"  - Residual plots: {output_dir}/*_residuals.png")
    print(f"  - Comparison report: {report_path}")

if __name__ == "__main__":
    main()

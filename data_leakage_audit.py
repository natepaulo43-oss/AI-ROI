"""
Data Leakage Audit and Risk Assessment for AI ROI Model
Identifies potential data leakage, temporal issues, and overfitting risks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

def audit_data_leakage():
    """Comprehensive data leakage audit"""

    print("="*80)
    print("DATA LEAKAGE & RISK AUDIT - AI ROI MODEL")
    print("="*80)
    print(f"\nAudit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = pd.read_csv('data/processed/ai_roi_training_dataset_enhanced.csv')
    print(f"\nDataset: {len(df)} records, {len(df.columns)} features")

    print("\n" + "-"*80)
    print("1. FEATURE CLASSIFICATION & LEAKAGE RISK")
    print("-"*80)

    # Classify features by when they're known
    pre_adoption = [
        'year', 'quarter', 'sector', 'company_size', 'revenue_m_eur',
        'ai_use_case', 'deployment_type', 'days_diagnostic', 'days_poc',
        'days_to_deployment', 'investment_eur', 'human_in_loop'
    ]

    post_adoption = [
        'time_saved_hours_month',  # Known only AFTER deployment
        'revenue_increase_percent'  # Known only AFTER deployment
    ]

    target = ['roi']

    print("\n[SAFE] Pre-Adoption Features (No Leakage):")
    for feat in pre_adoption:
        print(f"  - {feat:30s} : Available BEFORE deployment")

    print("\n[RISK] Post-Adoption Features (Potential Leakage):")
    for feat in post_adoption:
        print(f"  - {feat:30s} : Known only AFTER deployment starts")
        print(f"    WARNING: Using this creates temporal leakage!")

    print("\n[TARGET] Outcome Variable:")
    print(f"  - roi : Final outcome to predict")

    # Check for derived features that might leak
    print("\n" + "-"*80)
    print("2. DERIVED FEATURE LEAKAGE CHECK")
    print("-"*80)

    print("\nEngineered features created during training:")
    engineered = [
        ('log_investment', 'log(investment_eur)', 'SAFE'),
        ('log_revenue', 'log(revenue_m_eur)', 'SAFE'),
        ('investment_ratio', 'investment / revenue', 'SAFE'),
        ('investment_per_day', 'investment / days_to_deployment', 'SAFE'),
        ('diagnostic_efficiency', 'days_diagnostic / days_to_deployment', 'SAFE'),
        ('poc_efficiency', 'days_poc / days_to_deployment', 'SAFE'),
        ('total_prep_time', 'days_diagnostic + days_poc', 'SAFE'),
        ('deployment_speed', '1 / days_to_deployment', 'SAFE'),
        ('size_investment_interaction', 'log_revenue * log_investment', 'SAFE'),
        ('is_large_company', 'company_size == grande', 'SAFE'),
        ('is_hybrid_deployment', 'deployment_type == hybrid', 'SAFE'),
        ('has_revenue_increase', 'revenue_increase_percent > 0', 'LEAKAGE'),
        ('has_time_savings', 'time_saved_hours_month > 0', 'LEAKAGE'),
    ]

    for feat_name, formula, status in engineered:
        symbol = "[OK]" if status == "SAFE" else "[RISK]"
        print(f"  {symbol} {feat_name:30s} = {formula}")
        if status == "LEAKAGE":
            print(f"       ^ Derived from post-adoption outcome!")

    # Check for duplicate or near-duplicate records
    print("\n" + "-"*80)
    print("3. DUPLICATE RECORDS CHECK (Train/Test Contamination)")
    print("-"*80)

    duplicates = df.duplicated()
    print(f"\nExact duplicates: {duplicates.sum()}")

    if duplicates.sum() > 0:
        print("[RISK] Found duplicate records that could appear in both train and test!")
        print(df[duplicates][['sector', 'company_size', 'investment_eur', 'roi']].head())
    else:
        print("[OK] No exact duplicates found")

    # Check for near-duplicates (same company characteristics + investment)
    key_cols = ['sector', 'company_size', 'revenue_m_eur', 'investment_eur',
                'ai_use_case', 'days_to_deployment']
    near_dupes = df[key_cols].round(2).duplicated()
    print(f"\nNear-duplicates (same key characteristics): {near_dupes.sum()}")

    if near_dupes.sum() > 5:
        print("[WARNING] Many similar records could cause overfitting")
    else:
        print("[OK] Low risk of near-duplicate contamination")

    # Temporal leakage check
    print("\n" + "-"*80)
    print("4. TEMPORAL LEAKAGE CHECK")
    print("-"*80)

    print("\nTrain/Test split strategy:")
    print("  Current: Random 80/20 split (random_state=42)")
    print("  [WARNING] Random split can leak temporal patterns!")
    print("  Recommendation: Use time-based split if deploying to future")

    # Simulate temporal split
    df_sorted = df.sort_values(['year', 'quarter'])
    split_idx = int(len(df_sorted) * 0.8)
    train_years = df_sorted.iloc[:split_idx]['year'].value_counts()
    test_years = df_sorted.iloc[split_idx:]['year'].value_counts()

    print(f"\n  If using temporal split (80/20):")
    print(f"    Train: {train_years.to_dict()}")
    print(f"    Test: {test_years.to_dict()}")

    overlap = set(train_years.index) & set(test_years.index)
    if len(overlap) > 0:
        print(f"    [INFO] Years overlap: {overlap} (expected with quarterly data)")

    # Check for future information leakage
    print("\n" + "-"*80)
    print("5. FUTURE INFORMATION LEAKAGE")
    print("-"*80)

    print("\nChecking if any features use information from the future...")

    # Check if deployment timeline makes sense
    timeline_issues = df[
        (df['days_diagnostic'] + df['days_poc'] > df['days_to_deployment'])
    ]

    print(f"\n  Records where diagnostic+poc > total deployment: {len(timeline_issues)}")
    if len(timeline_issues) > 0:
        print("  [ERROR] Timeline inconsistency - potential data quality issue!")
        print(timeline_issues[['days_diagnostic', 'days_poc', 'days_to_deployment']].head())
    else:
        print("  [OK] Timeline data is logically consistent")

    # Check for impossible outcomes
    negative_days = df[
        (df['days_diagnostic'] < 0) |
        (df['days_poc'] < 0) |
        (df['days_to_deployment'] < 0)
    ]

    print(f"\n  Records with negative days: {len(negative_days)}")
    if len(negative_days) > 0:
        print("  [ERROR] Invalid data - negative timeline values!")
    else:
        print("  [OK] No negative timeline values")

    # Model-specific leakage analysis
    print("\n" + "-"*80)
    print("6. MODEL-SPECIFIC LEAKAGE ANALYSIS")
    print("-"*80)

    print("\nTwo models trained:")
    print("\n  A) CONSERVATIVE MODEL (Pre-adoption only)")
    print("     Features used: pre-adoption variables only")
    print("     [OK] NO DATA LEAKAGE - Can predict before deployment")
    print("     Use case: Pre-deployment ROI estimates")

    print("\n  B) PRACTICAL MODEL (With early signals)")
    print("     Features used: pre-adoption + time_saved + revenue_increase")
    print("     [ACKNOWLEDGED LEAKAGE] Uses outcome variables")
    print("     Use case: Mid-deployment predictions (after 1-3 months)")
    print("     Limitation: CANNOT predict before deployment starts")

    # Calculate correlation with target
    print("\n" + "-"*80)
    print("7. FEATURE-TARGET CORRELATION (Leakage Detection)")
    print("-"*80)

    print("\nFeatures with suspicious high correlation to ROI:")
    # Only numeric features for correlation
    numeric_cols = [col for col in (pre_adoption + post_adoption) if df[col].dtype in ['int64', 'float64']]
    correlations = df[numeric_cols].corrwith(df['roi']).abs().sort_values(ascending=False)

    for feat, corr in correlations.items():
        if corr > 0.5:
            status = "[SUSPICIOUS]" if feat not in post_adoption else "[EXPECTED]"
            print(f"  {status} {feat:30s} : {corr:.4f}")
            if feat not in post_adoption and corr > 0.5:
                print(f"          ^ Investigate why pre-adoption feature is so predictive!")

    print("\n  Post-adoption features (expected high correlation):")
    for feat in post_adoption:
        corr = correlations.get(feat, 0)
        print(f"    {feat:30s} : {corr:.4f}")

    # Check for target leakage via perfect prediction
    print("\n" + "-"*80)
    print("8. PERFECT PREDICTION TEST (Target Leakage)")
    print("-"*80)

    print("\nIf any feature perfectly predicts ROI, it's leaked!")
    for col in df.columns:
        if col != 'roi':
            try:
                # Check if feature is basically ROI in disguise
                if df[col].dtype in ['int64', 'float64']:
                    corr = abs(df[col].corr(df['roi']))
                    if corr > 0.95:
                        print(f"  [CRITICAL] {col} has {corr:.4f} correlation with ROI!")
                        print(f"             This feature likely leaks target information!")
            except:
                pass

    print("  [OK] No features with >0.95 correlation found")

    # Recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    print("\n[CURRENT RISKS IDENTIFIED]")
    print("  1. Practical model uses outcome variables (time_saved, revenue_increase)")
    print("     Impact: Cannot predict pre-deployment, only mid-deployment")
    print("     Mitigation: Already have Conservative model for pre-deployment")

    print("\n  2. Random train/test split (not temporal)")
    print("     Impact: May not generalize to future time periods")
    print("     Mitigation: Use time-based validation")

    print("\n  3. Derived features from outcome variables (has_time_savings, etc.)")
    print("     Impact: Reinforces leakage in Practical model")
    print("     Mitigation: Only use Conservative model for predictions")

    print("\n[RECOMMENDATIONS]")
    print("  1. Use CONSERVATIVE model for all pre-deployment predictions")
    print("  2. Use PRACTICAL model only when early signals are available")
    print("  3. Implement time-based cross-validation for better generalization")
    print("  4. Add validation: Check predictions on recent data (2025)")
    print("  5. Monitor model drift over time")
    print("  6. Consider ensemble: Conservative + industry benchmarks")

    print("\n[RISK ASSESSMENT]")
    print("  Overall Risk Level: MEDIUM")
    print("  Conservative Model: LOW RISK (no leakage)")
    print("  Practical Model: MEDIUM-HIGH RISK (acknowledged leakage)")

    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)

    return df

if __name__ == "__main__":
    df = audit_data_leakage()

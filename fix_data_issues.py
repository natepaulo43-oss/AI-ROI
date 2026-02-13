"""
Fix data quality issues found in audit
- Timeline inconsistencies (diagnostic + poc > deployment)
- Validate all data constraints
"""

import pandas as pd
import numpy as np

def fix_timeline_issues():
    """Fix timeline inconsistencies"""

    print("="*80)
    print("FIXING DATA QUALITY ISSUES")
    print("="*80)

    # Load data
    df = pd.read_csv('data/processed/ai_roi_training_dataset_enhanced.csv')
    print(f"\nOriginal dataset: {len(df)} records")

    # Identify inconsistent records
    inconsistent = df[
        (df['days_diagnostic'] + df['days_poc'] > df['days_to_deployment'])
    ]

    print(f"\nFound {len(inconsistent)} records with timeline issues:")
    print("\nBefore fixing:")
    print(inconsistent[['days_diagnostic', 'days_poc', 'days_to_deployment']].head())

    # Fix: Make deployment time = diagnostic + poc + actual deployment phase
    # Add 10% buffer for deployment phase (minimum 30 days)
    for idx in inconsistent.index:
        prep_time = df.loc[idx, 'days_diagnostic'] + df.loc[idx, 'days_poc']
        deployment_phase = max(30, int(prep_time * 0.3))  # 30% of prep time
        df.loc[idx, 'days_to_deployment'] = prep_time + deployment_phase

    # Verify fix
    still_inconsistent = df[
        (df['days_diagnostic'] + df['days_poc'] > df['days_to_deployment'])
    ]

    if len(still_inconsistent) == 0:
        print("\n[OK] All timeline issues fixed!")
    else:
        print(f"\n[ERROR] Still have {len(still_inconsistent)} issues")

    print("\nAfter fixing (sample):")
    print(df.loc[inconsistent.index[:5], ['days_diagnostic', 'days_poc', 'days_to_deployment']])

    # Additional validation
    print("\n" + "-"*80)
    print("ADDITIONAL VALIDATION")
    print("-"*80)

    # Check for negative values
    negative_checks = {
        'investment_eur': (df['investment_eur'] < 0).sum(),
        'revenue_m_eur': (df['revenue_m_eur'] < 0).sum(),
        'days_diagnostic': (df['days_diagnostic'] < 0).sum(),
        'days_poc': (df['days_poc'] < 0).sum(),
        'days_to_deployment': (df['days_to_deployment'] < 0).sum(),
        'time_saved_hours_month': (df['time_saved_hours_month'] < 0).sum(),
        'revenue_increase_percent': (df['revenue_increase_percent'] < 0).sum(),
    }

    print("\nNegative value check:")
    for col, count in negative_checks.items():
        status = "[OK]" if count == 0 else "[ERROR]"
        print(f"  {status} {col:30s}: {count} negative values")

    # Check for missing values
    print("\nMissing value check:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  [OK] No missing values")
    else:
        print(missing[missing > 0])

    # Check value ranges
    print("\nValue range validation:")

    checks = [
        ('investment_eur', df['investment_eur'].min() > 0, "Investment must be > 0"),
        ('revenue_m_eur', df['revenue_m_eur'].min() > 0, "Revenue must be > 0"),
        ('days_to_deployment', df['days_to_deployment'].min() >= 30, "Deployment >= 30 days"),
        ('roi', df['roi'].between(-100, 500).all(), "ROI between -100% and 500%"),
        ('human_in_loop', df['human_in_loop'].isin([0, 1]).all(), "Human in loop is 0 or 1"),
    ]

    for name, check, description in checks:
        status = "[OK]" if check else "[ERROR]"
        print(f"  {status} {description}")

    # Save cleaned data
    output_path = 'data/processed/ai_roi_training_dataset_cleaned.csv'
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Cleaned dataset saved to: {output_path}")
    print(f"     Records: {len(df)}")

    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFixed Issues:")
    print(f"  - Timeline inconsistencies: {len(inconsistent)} records fixed")
    print(f"  - Final dataset: {len(df)} clean records")
    print(f"\nNext Steps:")
    print(f"  1. Retrain models with cleaned data")
    print(f"  2. Update training script to use: {output_path}")
    print(f"  3. Implement validation checks in prediction pipeline")

    return df

if __name__ == "__main__":
    df = fix_timeline_issues()

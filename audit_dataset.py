import pandas as pd
import numpy as np
from pathlib import Path

def audit_dataset():
    print("=" * 80)
    print("DATASET AUDIT REPORT - AI ROI Prediction Model")
    print("=" * 80)
    
    df = pd.read_csv('data/raw/ai_roi_dataset_200_deployments.csv')
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATASET AUDIT REPORT - AI ROI Prediction Model")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Total columns: {len(df.columns)}")
    report_lines.append(f"Columns: {list(df.columns)}")
    report_lines.append("")
    
    print("\n1. DATASET OVERVIEW")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    report_lines.append("2. TARGET VARIABLE ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Target: roi_percent")
    report_lines.append(f"  Mean: {df['roi_percent'].mean():.2f}%")
    report_lines.append(f"  Median: {df['roi_percent'].median():.2f}%")
    report_lines.append(f"  Std Dev: {df['roi_percent'].std():.2f}%")
    report_lines.append(f"  Min: {df['roi_percent'].min():.2f}%")
    report_lines.append(f"  Max: {df['roi_percent'].max():.2f}%")
    report_lines.append(f"  Missing values: {df['roi_percent'].isna().sum()}")
    report_lines.append("")
    
    print("\n2. TARGET VARIABLE (roi_percent)")
    print(f"  Mean: {df['roi_percent'].mean():.2f}%")
    print(f"  Range: [{df['roi_percent'].min():.2f}, {df['roi_percent'].max():.2f}]")
    
    report_lines.append("3. CATEGORICAL FEATURES CARDINALITY")
    report_lines.append("-" * 80)
    categorical_cols = ['sector', 'company_size', 'ai_use_case', 'deployment_type']
    
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        value_counts = df[col].value_counts()
        report_lines.append(f"\n{col.upper()}:")
        report_lines.append(f"  Unique values: {unique_vals}")
        report_lines.append(f"  Value distribution:")
        for val, count in value_counts.items():
            pct = (count / len(df)) * 100
            report_lines.append(f"    {val}: {count} ({pct:.1f}%)")
    
    report_lines.append("")
    
    print("\n3. CATEGORICAL CARDINALITY")
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} unique values")
    
    report_lines.append("4. NUMERIC FEATURES CORRELATION WITH ROI")
    report_lines.append("-" * 80)
    
    numeric_cols = ['investment_eur', 'revenue_m_eur', 'days_to_deployment', 
                    'days_diagnostic', 'days_poc', 'human_in_loop']
    
    correlations = []
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df['roi_percent'])
            correlations.append((col, corr))
            report_lines.append(f"  {col}: r = {corr:.4f}")
    
    report_lines.append("")
    
    print("\n4. CORRELATIONS WITH ROI")
    for col, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {col}: r = {corr:.4f}")
    
    df['log_investment'] = np.log1p(df['investment_eur'])
    df['log_revenue'] = np.log1p(df['revenue_m_eur'])
    df['investment_ratio'] = df['investment_eur'] / df['revenue_m_eur']
    
    report_lines.append("5. ENGINEERED FEATURES CORRELATION WITH ROI")
    report_lines.append("-" * 80)
    engineered_cols = ['log_investment', 'log_revenue', 'investment_ratio']
    
    eng_correlations = []
    for col in engineered_cols:
        corr = df[col].corr(df['roi_percent'])
        eng_correlations.append((col, corr))
        report_lines.append(f"  {col}: r = {corr:.4f}")
    
    report_lines.append("")
    
    print("\n5. ENGINEERED FEATURES CORRELATIONS")
    for col, corr in eng_correlations:
        print(f"  {col}: r = {corr:.4f}")
    
    report_lines.append("6. POST-DEPLOYMENT FEATURES (SHOULD BE EXCLUDED)")
    report_lines.append("-" * 80)
    post_deployment = ['time_saved_hours_month', 'revenue_increase_percent', 
                       'annual_gain_eur', 'days_to_positive_roi']
    
    for col in post_deployment:
        if col in df.columns:
            corr = df[col].corr(df['roi_percent'])
            report_lines.append(f"  {col}: r = {corr:.4f} [EXCLUDED - POST-DEPLOYMENT]")
    
    report_lines.append("")
    
    report_lines.append("7. MISSING VALUES ANALYSIS")
    report_lines.append("-" * 80)
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    
    if len(missing_summary) > 0:
        for col, count in missing_summary.items():
            pct = (count / len(df)) * 100
            report_lines.append(f"  {col}: {count} missing ({pct:.1f}%)")
    else:
        report_lines.append("  No missing values detected in any column")
    
    report_lines.append("")
    
    print("\n6. MISSING VALUES")
    if len(missing_summary) > 0:
        print(f"  Found missing values in {len(missing_summary)} columns")
    else:
        print("  No missing values detected")
    
    report_lines.append("8. DATA QUALITY ISSUES")
    report_lines.append("-" * 80)
    
    issues = []
    
    if df['investment_eur'].min() <= 0:
        issues.append(f"  - investment_eur has non-positive values (min: {df['investment_eur'].min()})")
    
    if df['revenue_m_eur'].min() <= 0:
        issues.append(f"  - revenue_m_eur has non-positive values (min: {df['revenue_m_eur'].min()})")
    
    outliers_roi = df[(df['roi_percent'] < -50) | (df['roi_percent'] > 500)]
    if len(outliers_roi) > 0:
        issues.append(f"  - {len(outliers_roi)} extreme ROI outliers (< -50% or > 500%)")
    
    if len(issues) > 0:
        report_lines.extend(issues)
    else:
        report_lines.append("  No major data quality issues detected")
    
    report_lines.append("")
    
    report_lines.append("9. FEATURE STRENGTH SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append("Strong correlations (|r| > 0.3):")
    strong = [(c, r) for c, r in correlations + eng_correlations if abs(r) > 0.3]
    if strong:
        for col, corr in strong:
            report_lines.append(f"  - {col}: r = {corr:.4f}")
    else:
        report_lines.append("  NONE FOUND - This explains the low R²!")
    
    report_lines.append("")
    report_lines.append("Moderate correlations (0.2 < |r| <= 0.3):")
    moderate = [(c, r) for c, r in correlations + eng_correlations if 0.2 < abs(r) <= 0.3]
    if moderate:
        for col, corr in moderate:
            report_lines.append(f"  - {col}: r = {corr:.4f}")
    else:
        report_lines.append("  None")
    
    report_lines.append("")
    report_lines.append("Weak correlations (|r| <= 0.2):")
    weak = [(c, r) for c, r in correlations + eng_correlations if abs(r) <= 0.2]
    for col, corr in weak:
        report_lines.append(f"  - {col}: r = {corr:.4f}")
    
    report_lines.append("")
    
    report_lines.append("10. KEY FINDINGS")
    report_lines.append("-" * 80)
    
    all_corrs = correlations + eng_correlations
    max_corr = max([abs(r) for _, r in all_corrs])
    
    if max_corr < 0.2:
        report_lines.append("⚠️  CRITICAL: All pre-adoption features have WEAK correlations (|r| < 0.2)")
        report_lines.append("   This explains the R² collapse from 0.38 to 0.004")
        report_lines.append("   The original R²=0.38 was likely due to data leakage from post-deployment features")
    elif max_corr < 0.3:
        report_lines.append("⚠️  WARNING: No strong correlations found (max |r| = {:.4f})".format(max_corr))
        report_lines.append("   Pre-adoption features have limited predictive power")
    else:
        report_lines.append("✓  Some strong correlations exist - model should perform better than R²=0.004")
    
    report_lines.append("")
    
    total_dummies = sum([df[col].nunique() for col in categorical_cols])
    report_lines.append(f"Categorical feature complexity: {total_dummies} total unique categories")
    if total_dummies > 40:
        report_lines.append("⚠️  High cardinality may cause overfitting with small dataset (N=200)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF AUDIT REPORT")
    report_lines.append("=" * 80)
    
    Path('analysis').mkdir(exist_ok=True)
    with open('analysis/dataset_audit_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("\n✓ Audit complete! Report saved to: analysis/dataset_audit_report.txt")
    
    return df, correlations, eng_correlations

if __name__ == "__main__":
    audit_dataset()

"""
Main script to generate comprehensive AI ROI training dataset.
Combines web-scraped case studies with synthetic data generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Import our modules
from ai_roi_data_generator import AIROIDataGenerator
from web_scraper import AIROIWebScraper


def load_existing_data(file_path: str) -> pd.DataFrame:
    """Load existing training data if available"""
    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Loaded existing dataset: {len(df)} records")
        return df
    except FileNotFoundError:
        print("No existing dataset found - will create new one")
        return None


def merge_datasets(existing_df: pd.DataFrame = None,
                   case_studies_df: pd.DataFrame = None,
                   synthetic_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge all data sources"""
    dfs_to_merge = []

    if existing_df is not None:
        print(f"  • Existing data: {len(existing_df)} records")
        dfs_to_merge.append(existing_df)

    if case_studies_df is not None:
        print(f"  • Case studies: {len(case_studies_df)} records")
        dfs_to_merge.append(case_studies_df)

    if synthetic_df is not None:
        print(f"  • Synthetic data: {len(synthetic_df)} records")
        dfs_to_merge.append(synthetic_df)

    if not dfs_to_merge:
        raise ValueError("No data to merge!")

    merged = pd.concat(dfs_to_merge, ignore_index=True)
    print(f"\n[OK] Total merged records: {len(merged)}")

    return merged


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the dataset"""
    print("\nValidating dataset...")

    initial_count = len(df)

    # Check for required columns
    required_columns = [
        'year', 'quarter', 'sector', 'company_size', 'revenue_m_eur',
        'ai_use_case', 'deployment_type', 'days_diagnostic', 'days_poc',
        'days_to_deployment', 'investment_eur', 'roi', 'time_saved_hours_month',
        'revenue_increase_percent', 'human_in_loop'
    ]

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Validate data types and ranges
    assert df['year'].between(2020, 2030).all(), "Invalid years detected"
    assert df['quarter'].isin(['q1', 'q2', 'q3', 'q4']).all(), "Invalid quarters"
    assert df['company_size'].isin(['pme', 'eti', 'grande']).all(), "Invalid company sizes"
    assert (df['revenue_m_eur'] > 0).all(), "Invalid revenue values"
    assert (df['investment_eur'] > 0).all(), "Invalid investment values"
    assert (df['days_diagnostic'] >= 0).all(), "Invalid diagnostic days"
    assert (df['days_poc'] >= 0).all(), "Invalid POC days"
    assert (df['days_to_deployment'] >= 1).all(), "Invalid deployment days"
    assert (df['time_saved_hours_month'] >= 0).all(), "Invalid time savings"
    assert (df['revenue_increase_percent'] >= 0).all(), "Invalid revenue increase"
    assert df['human_in_loop'].isin([0, 1]).all(), "Invalid human_in_loop values"

    # Check for reasonable ROI range (-100% to 500%)
    outliers = df[(df['roi'] < -100) | (df['roi'] > 500)]
    if len(outliers) > 0:
        print(f"  Warning: {len(outliers)} ROI outliers detected (outside -100% to 500%)")
        df = df[(df['roi'] >= -100) & (df['roi'] <= 500)]

    final_count = len(df)
    if final_count < initial_count:
        print(f"  Removed {initial_count - final_count} invalid/duplicate records")

    print(f"[OK] Validation complete: {final_count} valid records")
    return df


def generate_report(df: pd.DataFrame, output_path: str):
    """Generate data quality report"""
    report = []
    report.append("="*80)
    report.append("AI ROI TRAINING DATASET - QUALITY REPORT")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nTotal Records: {len(df)}")

    report.append("\n" + "-"*80)
    report.append("FEATURE STATISTICS")
    report.append("-"*80)

    # Numeric features
    report.append(f"\nROI:")
    report.append(f"  Mean: {df['roi'].mean():.2f}%")
    report.append(f"  Median: {df['roi'].median():.2f}%")
    report.append(f"  Std Dev: {df['roi'].std():.2f}%")
    report.append(f"  Range: [{df['roi'].min():.2f}%, {df['roi'].max():.2f}%]")
    report.append(f"  Positive ROI: {(df['roi'] > 0).sum()} ({(df['roi'] > 0).sum()/len(df)*100:.1f}%)")
    report.append(f"  High ROI (>100%): {(df['roi'] > 100).sum()} ({(df['roi'] > 100).sum()/len(df)*100:.1f}%)")

    report.append(f"\nInvestment (EUR):")
    report.append(f"  Mean: €{df['investment_eur'].mean():,.0f}")
    report.append(f"  Median: €{df['investment_eur'].median():,.0f}")
    report.append(f"  Range: [€{df['investment_eur'].min():,.0f}, €{df['investment_eur'].max():,.0f}]")

    report.append(f"\nDeployment Timeline:")
    report.append(f"  Diagnostic phase: {df['days_diagnostic'].mean():.1f} days (avg)")
    report.append(f"  POC phase: {df['days_poc'].mean():.1f} days (avg)")
    report.append(f"  Total deployment: {df['days_to_deployment'].mean():.1f} days (avg)")

    report.append(f"\nOutcomes:")
    report.append(f"  Time savings: {df['time_saved_hours_month'].mean():.1f} hours/month (avg)")
    report.append(f"  Revenue increase: {df['revenue_increase_percent'].mean():.2f}% (avg)")
    report.append(f"  Records with time savings: {(df['time_saved_hours_month'] > 0).sum()} ({(df['time_saved_hours_month'] > 0).sum()/len(df)*100:.1f}%)")
    report.append(f"  Records with revenue increase: {(df['revenue_increase_percent'] > 0).sum()} ({(df['revenue_increase_percent'] > 0).sum()/len(df)*100:.1f}%)")

    # Categorical features
    report.append("\n" + "-"*80)
    report.append("CATEGORICAL DISTRIBUTIONS")
    report.append("-"*80)

    report.append(f"\nSectors: {df['sector'].nunique()} unique")
    for sector, count in df['sector'].value_counts().head(10).items():
        report.append(f"  {sector}: {count} ({count/len(df)*100:.1f}%)")

    report.append(f"\nCompany Sizes:")
    for size, count in df['company_size'].value_counts().items():
        report.append(f"  {size}: {count} ({count/len(df)*100:.1f}%)")

    report.append(f"\nAI Use Cases: {df['ai_use_case'].nunique()} unique")
    for use_case, count in df['ai_use_case'].value_counts().head(10).items():
        report.append(f"  {use_case}: {count} ({count/len(df)*100:.1f}%)")

    report.append(f"\nDeployment Types:")
    for dtype, count in df['deployment_type'].value_counts().items():
        report.append(f"  {dtype}: {count} ({count/len(df)*100:.1f}%)")

    report.append(f"\nYears:")
    for year, count in sorted(df['year'].value_counts().items()):
        report.append(f"  {year}: {count} ({count/len(df)*100:.1f}%)")

    report.append("\n" + "="*80)
    report.append("DATA SOURCES")
    report.append("="*80)
    report.append("\n• McKinsey: The State of AI in 2025")
    report.append("• Gartner: Gen AI Project Analysis 2024")
    report.append("• BCG: AI Adoption Study 2024")
    report.append("• Wharton: AI Adoption Report 2025")
    report.append("• Real case studies: Klarna, Alibaba, Walmart, Netflix, JPMorgan, etc.")
    report.append("• Industry statistics: 2022-2025 AI adoption data")

    report.append("\n" + "="*80)

    report_text = "\n".join(report)

    # Save report
    with open(output_path, 'w') as f:
        f.write(report_text)

    # Print to console
    print("\n" + report_text)

    return report_text


def main():
    """Main execution flow"""
    print("="*80)
    print("AI ROI TRAINING DATA GENERATION PIPELINE")
    print("="*80)
    print("\nThis script generates comprehensive AI ROI training data by:")
    print("1. Extracting real case studies from web sources")
    print("2. Generating synthetic data based on industry statistics")
    print("3. Merging and validating the complete dataset\n")

    # Paths
    project_root = Path(__file__).parent.parent.parent
    existing_data_path = project_root / 'data' / 'processed' / 'ai_roi_modeling_dataset.csv'
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load existing data
    print("\n[1/5] Loading existing data...")
    existing_df = load_existing_data(existing_data_path)

    # Step 2: Extract case studies
    print("\n[2/5] Extracting case studies from web sources...")
    scraper = AIROIWebScraper(rate_limit_seconds=1.0)
    case_studies_df = scraper.scrape_and_save(
        output_file=str(output_dir / 'ai_roi_case_studies.csv')
    )

    # Step 3: Generate synthetic data
    print("\n[3/5] Generating synthetic data based on industry research...")
    generator = AIROIDataGenerator()
    synthetic_df = generator.generate_dataset(n_records=200)

    # Step 4: Merge datasets
    print("\n[4/5] Merging datasets...")
    print("Sources to merge:")
    final_df = merge_datasets(
        existing_df=existing_df,
        case_studies_df=case_studies_df,
        synthetic_df=synthetic_df
    )

    # Step 5: Validate and save
    print("\n[5/5] Validating and saving final dataset...")
    final_df = validate_data(final_df)

    # Save final dataset
    output_file = output_dir / 'ai_roi_training_dataset_enhanced.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n[OK] Final dataset saved to: {output_file}")

    # Generate report
    report_file = output_dir / 'data_quality_report.txt'
    generate_report(final_df, report_file)
    print(f"[OK] Quality report saved to: {report_file}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nReady for model retraining with {len(final_df)} records!")
    print(f"Dataset location: {output_file}")

    return final_df


if __name__ == "__main__":
    try:
        df = main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

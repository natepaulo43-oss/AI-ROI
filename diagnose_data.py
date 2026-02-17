import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/ai_roi_full_combined_cleaned.csv')

print("Dataset Statistics:")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nROI Statistics:")
print(df['roi'].describe())

print(f"\nROI outliers (>500): {(df['roi'] > 500).sum()}")
print(f"ROI negative: {(df['roi'] < 0).sum()}")

print("\nSample of extreme ROI values:")
print(df.nlargest(5, 'roi')[['sector', 'company_size', 'investment_eur', 'roi']])

print("\nSample of negative ROI values:")
print(df[df['roi'] < 0][['sector', 'company_size', 'investment_eur', 'roi']].head())

print("\nColumn names:")
print(df.columns.tolist())

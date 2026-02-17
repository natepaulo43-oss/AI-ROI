import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv('data/processed/ai_roi_full_combined_cleaned.csv')
print(f"Original shape: {df.shape}")

invalid_mask = df['days_diagnostic'] + df['days_poc'] > df['days_to_deployment']
print(f"Rows with timeline issues: {invalid_mask.sum()}")

for idx in df[invalid_mask].index:
    total_prep = df.loc[idx, 'days_diagnostic'] + df.loc[idx, 'days_poc']
    df.loc[idx, 'days_to_deployment'] = total_prep + np.random.randint(5, 30)

invalid_after = df['days_diagnostic'] + df['days_poc'] > df['days_to_deployment']
print(f"Rows with timeline issues after fix: {invalid_after.sum()}")

df.to_csv('data/processed/ai_roi_full_combined_cleaned.csv', index=False)
print("Dataset fixed and saved!")

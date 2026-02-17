import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/processed/515.csv')

print("=" * 80)
print("COMPANY SIZE GROUPING ANALYSIS")
print("=" * 80)

# 1. Distribution Analysis
print("\n1. CURRENT DISTRIBUTION (3 categories)")
print("-" * 40)
size_dist = df['company_size'].value_counts().sort_index()
print(size_dist)
print(f"\nTotal records: {len(df)}")
for size in ['pme', 'eti', 'grande']:
    pct = (size_dist[size] / len(df)) * 100
    print(f"{size}: {size_dist[size]} ({pct:.1f}%)")

# 2. ROI Statistics by Company Size
print("\n2. ROI STATISTICS BY COMPANY SIZE")
print("-" * 40)
roi_stats = df.groupby('company_size')['roi'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(roi_stats.round(2))

# 3. Revenue Distribution by Company Size
print("\n3. REVENUE DISTRIBUTION BY COMPANY SIZE")
print("-" * 40)
revenue_stats = df.groupby('company_size')['revenue_m_eur'].agg(['mean', 'median', 'min', 'max'])
print(revenue_stats.round(2))

# 4. Investment Patterns
print("\n4. INVESTMENT PATTERNS BY COMPANY SIZE")
print("-" * 40)
investment_stats = df.groupby('company_size')['investment_eur'].agg(['mean', 'median', 'min', 'max'])
print(investment_stats.round(0))

# 5. Check for natural groupings in revenue
print("\n5. REVENUE QUARTILES (Natural Groupings)")
print("-" * 40)
quartiles = df['revenue_m_eur'].quantile([0, 0.25, 0.5, 0.75, 1.0])
print(quartiles)

# 6. Alternative grouping analysis
print("\n6. ALTERNATIVE GROUPING COMPARISON")
print("-" * 40)

# Create alternative 4-category grouping based on revenue quartiles
df['size_4cat'] = pd.qcut(df['revenue_m_eur'], q=4, labels=['micro', 'small', 'medium', 'large'])
print("\nAlternative 4-category grouping (revenue quartiles):")
print(df['size_4cat'].value_counts().sort_index())

# Compare ROI variance
print("\n7. ROI VARIANCE COMPARISON")
print("-" * 40)
print(f"Current 3-category variance: {df.groupby('company_size')['roi'].var().mean():.2f}")
print(f"Alternative 4-category variance: {df.groupby('size_4cat')['roi'].var().mean():.2f}")

# 8. Statistical significance test
print("\n8. BETWEEN-GROUP VARIANCE (Higher = Better Separation)")
print("-" * 40)
from scipy import stats

# ANOVA test for current grouping
f_stat_3, p_value_3 = stats.f_oneway(
    df[df['company_size'] == 'pme']['roi'],
    df[df['company_size'] == 'eti']['roi'],
    df[df['company_size'] == 'grande']['roi']
)
print(f"Current 3-category ANOVA: F={f_stat_3:.2f}, p={p_value_3:.4f}")

# ANOVA test for alternative grouping
f_stat_4, p_value_4 = stats.f_oneway(
    df[df['size_4cat'] == 'micro']['roi'],
    df[df['size_4cat'] == 'small']['roi'],
    df[df['size_4cat'] == 'medium']['roi'],
    df[df['size_4cat'] == 'large']['roi']
)
print(f"Alternative 4-category ANOVA: F={f_stat_4:.2f}, p={p_value_4:.4f}")

# 9. Model feature importance check
print("\n9. RECOMMENDATION")
print("-" * 40)
if p_value_3 < 0.05:
    print("✓ Current 3-category grouping shows SIGNIFICANT differences in ROI")
else:
    print("✗ Current 3-category grouping does NOT show significant differences")

print(f"\nBalance check:")
min_group = size_dist.min()
max_group = size_dist.max()
balance_ratio = min_group / max_group
print(f"  Smallest group: {min_group} records")
print(f"  Largest group: {max_group} records")
print(f"  Balance ratio: {balance_ratio:.2f} (>0.3 is good)")

if balance_ratio > 0.3 and p_value_3 < 0.05:
    print("\n✓ VERDICT: Current 3-category grouping is OPTIMAL")
    print("  - Well-balanced distribution")
    print("  - Statistically significant ROI differences")
    print("  - Aligns with industry standards (employee count)")
else:
    print("\n⚠ VERDICT: Consider alternative grouping")

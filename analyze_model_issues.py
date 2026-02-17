import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE MODEL DIAGNOSTICS")
print("=" * 80)

df = pd.read_csv('data/processed/ai_roi_full_combined_cleaned.csv')
print(f"\n1. Dataset: {df.shape}")

roi_q1 = df['roi'].quantile(0.05)
roi_q99 = df['roi'].quantile(0.95)
df_filtered = df[(df['roi'] >= roi_q1) & (df['roi'] <= roi_q99)].copy()
print(f"   After outlier removal: {df_filtered.shape}")

y = df_filtered['roi'].copy()
df_work = df_filtered.drop(columns=['roi'])

print("\n2. FEATURE ENGINEERING")
df_work['log_investment'] = np.log1p(df_work['investment_eur'])
df_work['log_revenue'] = np.log1p(df_work['revenue_m_eur'])
df_work['investment_ratio'] = df_work['investment_eur'] / (df_work['revenue_m_eur'] * 1_000_000 + 1)
df_work['investment_per_day'] = df_work['investment_eur'] / (df_work['days_to_deployment'] + 1)
df_work['total_prep_time'] = df_work['days_diagnostic'] + df_work['days_poc']
df_work['deployment_speed'] = 1 / (df_work['days_to_deployment'] + 1)
df_work['is_large_company'] = (df_work['company_size'] == 'grande').astype(int)
df_work['human_in_loop'] = df_work['human_in_loop'].astype(int)

numeric_features = [
    'log_investment', 'log_revenue', 'investment_ratio', 'investment_per_day',
    'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
    'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year'
]
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']

print("\n3. DATA LEAKAGE CHECK")
print("   Checking for features that leak target information...")

leakage_suspects = ['time_saved_hours_month', 'revenue_increase_percent']
for feat in leakage_suspects:
    corr = df_work[feat].corr(y)
    print(f"   - {feat}: correlation with ROI = {corr:.4f}")
    if abs(corr) > 0.7:
        print(f"     ⚠️  HIGH CORRELATION - Potential data leakage!")
    elif abs(corr) > 0.4:
        print(f"     ⚠️  MODERATE CORRELATION - May be post-deployment signal")

print("\n4. FEATURE CORRELATION ANALYSIS")
X_numeric = df_work[numeric_features].copy()
correlations = []
for feat in numeric_features:
    corr = X_numeric[feat].corr(y)
    correlations.append((feat, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("   Top 10 features by correlation with ROI:")
for feat, corr in correlations[:10]:
    print(f"   {feat:35s}: {corr:7.4f}")

print("\n5. MULTICOLLINEARITY CHECK")
corr_matrix = X_numeric.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print("   ⚠️  High multicollinearity detected:")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"   - {feat1} <-> {feat2}: {corr:.4f}")
else:
    print("   ✓ No severe multicollinearity (threshold: 0.8)")

print("\n6. DIMENSIONALITY ANALYSIS")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_transformed = preprocessor.fit_transform(df_work[numeric_features + categorical_features])
print(f"   Original features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")
print(f"   After one-hot encoding: {X_transformed.shape[1]} features")
print(f"   Samples: {X_transformed.shape[0]}")
print(f"   Feature-to-sample ratio: 1:{X_transformed.shape[0]/X_transformed.shape[1]:.1f}")

if X_transformed.shape[1] > X_transformed.shape[0] / 10:
    print("   ⚠️  High dimensionality relative to sample size!")
    print("   → Dimensionality reduction recommended")

print("\n7. MUTUAL INFORMATION SCORES")
mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
mi_ranking = sorted(zip(numeric_features, mi_scores), key=lambda x: x[1], reverse=True)
print("   Top features by mutual information:")
for feat, score in mi_ranking[:10]:
    print(f"   {feat:35s}: {score:.4f}")

print("\n8. VARIANCE ANALYSIS")
print(f"   Target (ROI) variance: {y.var():.2f}")
print(f"   Target std dev: {y.std():.2f}")
print(f"   Coefficient of variation: {(y.std() / y.mean()):.2f}")

if y.std() / y.mean() > 1.0:
    print("   ⚠️  High coefficient of variation (>1.0)")
    print("   → Target is highly variable relative to mean")
    print("   → Consider log transformation or classification")

print("\n9. PCA ANALYSIS")
pca = PCA()
X_pca = pca.fit_transform(X_transformed)
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

print(f"   Components for 90% variance: {n_components_90}/{X_transformed.shape[1]}")
print(f"   Components for 95% variance: {n_components_95}/{X_transformed.shape[1]}")
print(f"   First 10 components explain: {cumsum_variance[9]:.2%} of variance")

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

issues = []
recommendations = []

if any(abs(corr) > 0.7 for _, corr in correlations if _ in leakage_suspects):
    issues.append("⚠️  Potential data leakage detected")
    recommendations.append("Remove or carefully validate post-deployment features")

if high_corr_pairs:
    issues.append("⚠️  Multicollinearity present")
    recommendations.append("Remove redundant features or use PCA")

if X_transformed.shape[1] > X_transformed.shape[0] / 10:
    issues.append("⚠️  High dimensionality (curse of dimensionality)")
    recommendations.append(f"Reduce to {n_components_90}-{n_components_95} components via PCA")

if y.std() / y.mean() > 1.0:
    issues.append("⚠️  High target variance")
    recommendations.append("Consider log(ROI) transformation or classification approach")

if len(df_filtered) < 500:
    issues.append("⚠️  Small dataset size")
    recommendations.append("Collect more data or use simpler models")

print("\nISSUES FOUND:")
for issue in issues:
    print(f"  {issue}")

print("\nRECOMMENDATIONS:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

print("\n" + "=" * 80)

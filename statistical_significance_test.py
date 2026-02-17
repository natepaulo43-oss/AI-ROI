import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, permutation_test_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STATISTICAL SIGNIFICANCE TESTING - MODEL PERFORMANCE")
print("=" * 80)

# Load data
df = pd.read_csv('data/processed/515.csv')
print(f"\n1. Dataset loaded: {df.shape}")

roi_q1 = df['roi'].quantile(0.05)
roi_q99 = df['roi'].quantile(0.95)
df_filtered = df[(df['roi'] >= roi_q1) & (df['roi'] <= roi_q99)].copy()

# Binary classification
q67 = df_filtered['roi'].quantile(0.67)
df_filtered['roi_binary'] = (df_filtered['roi'] >= q67).astype(int)
y = df_filtered['roi_binary'].copy()
df_work = df_filtered.drop(columns=['roi', 'roi_binary'])

# Feature engineering
df_work['log_investment'] = np.log1p(df_work['investment_eur'])
df_work['log_revenue'] = np.log1p(df_work['revenue_m_eur'])
df_work['investment_per_day'] = df_work['investment_eur'] / (df_work['days_to_deployment'] + 1)
df_work['total_prep_time'] = df_work['days_diagnostic'] + df_work['days_poc']
df_work['deployment_speed'] = 1 / (df_work['days_to_deployment'] + 1)
df_work['is_large_company'] = (df_work['company_size'] == 'grande').astype(int)
df_work['human_in_loop'] = df_work['human_in_loop'].astype(int)
df_work['revenue_investment_ratio'] = df_work['revenue_m_eur'] / (df_work['investment_eur'] / 1000000 + 1)
df_work['time_efficiency'] = df_work['time_saved_hours_month'] / (df_work['total_prep_time'] + 1)
df_work['revenue_time_interaction'] = df_work['revenue_increase_percent'] * df_work['time_saved_hours_month']

numeric_features = [
    'log_investment', 'log_revenue', 'investment_per_day', 'total_prep_time',
    'deployment_speed', 'time_saved_hours_month', 'revenue_increase_percent',
    'is_large_company', 'human_in_loop', 'year', 'revenue_investment_ratio',
    'time_efficiency', 'revenue_time_interaction'
]
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']

X = df_work[numeric_features + categorical_features].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Build model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=2, random_state=42, n_jobs=-1, eval_metric='logloss'
    ))
])

print("\n2. Training model and calculating statistics...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ============================================================================
# TEST 1: Cross-Validation with Confidence Intervals
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
cv_se = cv_std / np.sqrt(len(cv_scores))

# 95% confidence interval
confidence_level = 0.95
degrees_freedom = len(cv_scores) - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_value * cv_se
ci_lower = cv_mean - margin_of_error
ci_upper = cv_mean + margin_of_error

print(f"\n   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean CV Accuracy: {cv_mean:.4f} ({cv_mean*100:.2f}%)")
print(f"   Standard Deviation: {cv_std:.4f}")
print(f"   Standard Error: {cv_se:.4f}")
print(f"   95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"   95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# ============================================================================
# TEST 2: Permutation Test (Null Hypothesis Testing)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: PERMUTATION TEST (Null Hypothesis: Model = Random)")
print("=" * 80)

print("\n   Running permutation test (100 permutations)...")
print("   This tests if model performance is significantly better than random chance")

# Permutation test
score, permutation_scores, pvalue = permutation_test_score(
    model, X_train, y_train, scoring='accuracy', cv=cv, n_permutations=100, n_jobs=-1, random_state=42
)

print(f"\n   Model Score: {score:.4f} ({score*100:.2f}%)")
print(f"   Permutation Scores Mean: {permutation_scores.mean():.4f}")
print(f"   Permutation Scores Std: {permutation_scores.std():.4f}")
print(f"   P-Value: {pvalue:.6f}")

if pvalue < 0.001:
    print(f"\n   ✓ HIGHLY SIGNIFICANT (p < 0.001)")
    print(f"   → Model is significantly better than random chance")
elif pvalue < 0.01:
    print(f"\n   ✓ VERY SIGNIFICANT (p < 0.01)")
    print(f"   → Model is significantly better than random chance")
elif pvalue < 0.05:
    print(f"\n   ✓ SIGNIFICANT (p < 0.05)")
    print(f"   → Model is significantly better than random chance")
else:
    print(f"\n   ✗ NOT SIGNIFICANT (p ≥ 0.05)")
    print(f"   → Cannot reject null hypothesis")

# ============================================================================
# TEST 3: Binomial Test (Test vs Baseline)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: BINOMIAL TEST (vs Random Baseline)")
print("=" * 80)

# Test against random baseline (50% for binary classification)
n_test = len(y_test)
n_correct = (y_pred == y_test).sum()
baseline_prob = 0.5

# Binomial test
binomial_result = stats.binomtest(n_correct, n_test, baseline_prob, alternative='greater')
binomial_pvalue = binomial_result.pvalue

print(f"\n   Test samples: {n_test}")
print(f"   Correct predictions: {n_correct}")
print(f"   Observed accuracy: {n_correct/n_test:.4f} ({n_correct/n_test*100:.2f}%)")
print(f"   Baseline (random): {baseline_prob:.4f} ({baseline_prob*100:.2f}%)")
print(f"   P-Value (one-tailed): {binomial_pvalue:.6f}")

if binomial_pvalue < 0.001:
    print(f"\n   ✓ HIGHLY SIGNIFICANT (p < 0.001)")
    print(f"   → Model significantly outperforms random guessing")
elif binomial_pvalue < 0.01:
    print(f"\n   ✓ VERY SIGNIFICANT (p < 0.01)")
    print(f"   → Model significantly outperforms random guessing")
elif binomial_pvalue < 0.05:
    print(f"\n   ✓ SIGNIFICANT (p < 0.05)")
    print(f"   → Model significantly outperforms random guessing")
else:
    print(f"\n   ✗ NOT SIGNIFICANT (p ≥ 0.05)")

# ============================================================================
# TEST 4: McNemar's Test (Paired Test)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: EFFECT SIZE (Cohen's h)")
print("=" * 80)

# Calculate effect size (Cohen's h for proportions)
observed_prop = test_accuracy
baseline_prop = 0.5

# Cohen's h
cohens_h = 2 * (np.arcsin(np.sqrt(observed_prop)) - np.arcsin(np.sqrt(baseline_prop)))

print(f"\n   Observed accuracy: {observed_prop:.4f}")
print(f"   Baseline accuracy: {baseline_prop:.4f}")
print(f"   Cohen's h: {cohens_h:.4f}")

if abs(cohens_h) >= 0.8:
    effect_size = "LARGE"
elif abs(cohens_h) >= 0.5:
    effect_size = "MEDIUM"
elif abs(cohens_h) >= 0.2:
    effect_size = "SMALL"
else:
    effect_size = "NEGLIGIBLE"

print(f"   Effect Size: {effect_size}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE SUMMARY")
print("=" * 80)

print("\n📊 MODEL PERFORMANCE:")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"   95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

print("\n📈 STATISTICAL TESTS:")
print(f"   1. Permutation Test: p = {pvalue:.6f} {'✓ SIGNIFICANT' if pvalue < 0.05 else '✗ NOT SIGNIFICANT'}")
print(f"   2. Binomial Test: p = {binomial_pvalue:.6f} {'✓ SIGNIFICANT' if binomial_pvalue < 0.05 else '✗ NOT SIGNIFICANT'}")
print(f"   3. Effect Size (Cohen's h): {cohens_h:.4f} ({effect_size})")

print("\n🎯 INTERPRETATION:")
if pvalue < 0.05 and binomial_pvalue < 0.05:
    print("   ✓ Model performance is STATISTICALLY SIGNIFICANT")
    print("   ✓ Model significantly outperforms random chance")
    print("   ✓ Results are reliable and not due to chance")
    
    if test_accuracy > 0.65:
        print("   ✓ Model has GOOD practical performance (>65%)")
    elif test_accuracy > 0.60:
        print("   ✓ Model has MODERATE practical performance (60-65%)")
    else:
        print("   ⚠️  Model has LIMITED practical performance (<60%)")
else:
    print("   ✗ Model performance is NOT statistically significant")
    print("   ✗ Cannot confidently say model is better than random")

print("\n📋 CONFIDENCE IN PREDICTIONS:")
if cv_std < 0.05:
    print("   ✓ LOW VARIANCE: Model is stable across different data splits")
elif cv_std < 0.10:
    print("   ⚠️  MODERATE VARIANCE: Some variability in performance")
else:
    print("   ✗ HIGH VARIANCE: Model performance is unstable")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if pvalue < 0.001 and binomial_pvalue < 0.001:
    print("\n✓ HIGHLY SIGNIFICANT: Model is production-ready")
    print("  → P-values < 0.001 indicate very strong evidence")
    print("  → Model reliably outperforms random chance")
    print("  → Safe to deploy for decision support")
elif pvalue < 0.05 and binomial_pvalue < 0.05:
    print("\n✓ SIGNIFICANT: Model is suitable for production")
    print("  → P-values < 0.05 indicate strong evidence")
    print("  → Model outperforms random chance")
    print("  → Deploy with appropriate uncertainty communication")
else:
    print("\n⚠️  NOT SIGNIFICANT: Use with caution")
    print("  → P-values ≥ 0.05 indicate weak evidence")
    print("  → Model may not be better than random")
    print("  → Consider collecting more data")

print("\n" + "=" * 80)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED MODEL IMPROVEMENTS - MAXIMIZING PERFORMANCE")
print("=" * 80)

# Load data
df = pd.read_csv('data/processed/515.csv')
print(f"\n1. Dataset loaded: {df.shape}")

roi_q1 = df['roi'].quantile(0.05)
roi_q99 = df['roi'].quantile(0.95)
df_filtered = df[(df['roi'] >= roi_q1) & (df['roi'] <= roi_q99)].copy()
print(f"   After outlier removal: {df_filtered.shape}")

# Define thresholds
q33 = df_filtered['roi'].quantile(0.33)
q67 = df_filtered['roi'].quantile(0.67)

def classify_roi(roi):
    if roi < q33:
        return 0  # Low
    elif roi < q67:
        return 1  # Medium
    else:
        return 2  # High

df_filtered['roi_class'] = df_filtered['roi'].apply(classify_roi)
y = df_filtered['roi_class'].copy()
df_work = df_filtered.drop(columns=['roi', 'roi_class'])

print("\n2. ADVANCED FEATURE ENGINEERING")

# Basic features
df_work['log_investment'] = np.log1p(df_work['investment_eur'])
df_work['log_revenue'] = np.log1p(df_work['revenue_m_eur'])
df_work['investment_per_day'] = df_work['investment_eur'] / (df_work['days_to_deployment'] + 1)
df_work['total_prep_time'] = df_work['days_diagnostic'] + df_work['days_poc']
df_work['deployment_speed'] = 1 / (df_work['days_to_deployment'] + 1)
df_work['is_large_company'] = (df_work['company_size'] == 'grande').astype(int)
df_work['human_in_loop'] = df_work['human_in_loop'].astype(int)

# NEW: Advanced interaction features
df_work['revenue_investment_ratio'] = df_work['revenue_m_eur'] / (df_work['investment_eur'] / 1000000 + 1)
df_work['time_efficiency'] = df_work['time_saved_hours_month'] / (df_work['total_prep_time'] + 1)
df_work['revenue_time_interaction'] = df_work['revenue_increase_percent'] * df_work['time_saved_hours_month']
df_work['investment_speed_ratio'] = df_work['log_investment'] * df_work['deployment_speed']
df_work['prep_deployment_ratio'] = df_work['total_prep_time'] / (df_work['days_to_deployment'] + 1)

# NEW: Polynomial features for key numeric variables
df_work['log_investment_squared'] = df_work['log_investment'] ** 2
df_work['time_saved_squared'] = df_work['time_saved_hours_month'] ** 2
df_work['revenue_increase_squared'] = df_work['revenue_increase_percent'] ** 2

# NEW: Binned features
df_work['investment_bin'] = pd.qcut(df_work['investment_eur'], q=4, labels=['very_low', 'low', 'high', 'very_high'], duplicates='drop')
df_work['revenue_bin'] = pd.qcut(df_work['revenue_m_eur'], q=4, labels=['very_low', 'low', 'high', 'very_high'], duplicates='drop')

# NEW: Domain-specific features
df_work['has_revenue_impact'] = (df_work['revenue_increase_percent'] > 0).astype(int)
df_work['has_time_savings'] = (df_work['time_saved_hours_month'] > 0).astype(int)
df_work['has_both_benefits'] = ((df_work['revenue_increase_percent'] > 0) & (df_work['time_saved_hours_month'] > 0)).astype(int)

numeric_features = [
    'log_investment', 'log_revenue', 'investment_per_day',
    'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
    'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year',
    'revenue_investment_ratio', 'time_efficiency', 'revenue_time_interaction',
    'investment_speed_ratio', 'prep_deployment_ratio',
    'log_investment_squared', 'time_saved_squared', 'revenue_increase_squared',
    'has_revenue_impact', 'has_time_savings', 'has_both_benefits'
]

categorical_features = [
    'sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter',
    'investment_bin', 'revenue_bin'
]

X = df_work[numeric_features + categorical_features].copy()
print(f"   Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
print(f"   NEW features added: 14 (interactions, polynomials, bins, domain)")

print("\n3. Train/test split (stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# ============================================================================
# STRATEGY 1: Optimized Single Models
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 1: OPTIMIZED SINGLE MODELS")
print("=" * 80)

models_optimized = {
    'XGBoost_Optimized': XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    ),
    'RandomForest_Optimized': RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=6,
        min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    'LightGBM_Optimized': LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.03,
        num_leaves=50, min_child_samples=6, subsample=0.9,
        colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    ),
    'GradientBoosting_Optimized': GradientBoostingClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        min_samples_split=8, subsample=0.9, max_features='sqrt',
        random_state=42
    )
}

results_optimized = {}
for name, clf in models_optimized.items():
    print(f"\n   Training {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    results_optimized[name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': pipeline,
        'y_pred': y_pred
    }
    
    print(f"      Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ============================================================================
# STRATEGY 2: Ensemble Methods (Voting & Stacking)
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 2: ENSEMBLE METHODS")
print("=" * 80)

# Voting Classifier (Soft)
print("\n   Training Voting Ensemble (Soft)...")
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.03,
                             min_child_weight=2, subsample=0.9, random_state=42,
                             n_jobs=-1, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=6,
                                     class_weight='balanced', random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.03,
                              num_leaves=50, class_weight='balanced',
                              random_state=42, verbose=-1, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=500, max_depth=7, learning_rate=0.03,
                                         random_state=42))
    ],
    voting='soft',
    weights=[2, 1, 2, 1]  # Give more weight to XGBoost and LightGBM
)

voting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

voting_pipeline.fit(X_train, y_train)
voting_pred = voting_pipeline.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
voting_f1 = precision_recall_fscore_support(y_test, voting_pred, average='weighted')[2]
voting_cv = cross_val_score(voting_pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"   Voting Accuracy: {voting_acc:.4f} | F1: {voting_f1:.4f} | CV: {voting_cv.mean():.4f}±{voting_cv.std():.4f}")

# Stacking Classifier
print("\n   Training Stacking Ensemble...")
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.03,
                             random_state=42, n_jobs=-1, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=400, max_depth=15, class_weight='balanced',
                                     random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=400, max_depth=10, learning_rate=0.03,
                              class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=400, max_depth=7, learning_rate=0.03,
                                         random_state=42))
    ],
    final_estimator=XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42, eval_metric='mlogloss'),
    cv=5,
    n_jobs=-1
)

stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

print("   (This may take a few minutes...)")
stacking_pipeline.fit(X_train, y_train)
stacking_pred = stacking_pipeline.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)
stacking_f1 = precision_recall_fscore_support(y_test, stacking_pred, average='weighted')[2]
print(f"   Stacking Accuracy: {stacking_acc:.4f} | F1: {stacking_f1:.4f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS - ALL STRATEGIES")
print("=" * 80)

all_results = {
    'Baseline (GradientBoosting)': {'accuracy': 0.5161, 'f1': 0.5170, 'cv_mean': 0.5013},
    **{k: {'accuracy': v['accuracy'], 'f1': v['f1'], 'cv_mean': v['cv_mean']} for k, v in results_optimized.items()},
    'Voting_Ensemble': {'accuracy': voting_acc, 'f1': voting_f1, 'cv_mean': voting_cv.mean()},
    'Stacking_Ensemble': {'accuracy': stacking_acc, 'f1': stacking_f1, 'cv_mean': np.nan}
}

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\nRANKING (by Test Accuracy):")
print(f"{'Rank':<6}{'Model':<35}{'Accuracy':<12}{'F1-Score':<12}{'CV Acc':<12}{'Improvement':<12}")
print("-" * 87)

baseline_acc = 0.5161
for rank, (name, res) in enumerate(sorted_results, 1):
    improvement = ((res['accuracy'] - baseline_acc) / baseline_acc) * 100
    cv_str = f"{res['cv_mean']:.4f}" if not np.isnan(res['cv_mean']) else "N/A"
    marker = "★" if rank == 1 else " "
    print(f"{rank:<6}{marker} {name:<33}{res['accuracy']:<12.4f}{res['f1']:<12.4f}{cv_str:<12}{improvement:+.1f}%")

# Detailed results for best model
best_name, best_res = sorted_results[0]
print("\n" + "=" * 80)
print(f"BEST MODEL: {best_name}")
print("=" * 80)
print(f"\nAccuracy: {best_res['accuracy']:.4f} ({best_res['accuracy']*100:.2f}%)")
print(f"F1-Score: {best_res['f1']:.4f}")
print(f"Improvement over baseline: {((best_res['accuracy'] - baseline_acc) / baseline_acc) * 100:+.1f}%")

# Get predictions for best model
if best_name in results_optimized:
    best_pred = results_optimized[best_name]['y_pred']
    best_model = results_optimized[best_name]['pipeline']
elif best_name == 'Voting_Ensemble':
    best_pred = voting_pred
    best_model = voting_pipeline
elif best_name == 'Stacking_Ensemble':
    best_pred = stacking_pred
    best_model = stacking_pipeline
else:
    best_pred = None
    best_model = None

if best_pred is not None:
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, best_pred)
    class_names = ['Low', 'Medium', 'High']
    print(f"{'':12s} {'Pred Low':>12s} {'Pred Med':>12s} {'Pred High':>12s}")
    for i, name in enumerate(class_names):
        print(f"Actual {name:6s} {cm[i,0]:>12d} {cm[i,1]:>12d} {cm[i,2]:>12d}")
    
    print(f"\nPer-Class Performance:")
    precision, recall, f1, support = precision_recall_fscore_support(y_test, best_pred)
    for i, name in enumerate(class_names):
        print(f"  {name:8s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

# Save best model
model_dir = 'backend/models'
os.makedirs(model_dir, exist_ok=True)

if best_model is not None:
    best_path = os.path.join(model_dir, 'roi_classifier_advanced.pkl')
    joblib.dump(best_model, best_path)
    print(f"\n✓ Best model saved: {best_path}")
    
    metadata = {
        'model_name': best_name,
        'accuracy': best_res['accuracy'],
        'f1': best_res['f1'],
        'cv_mean': best_res['cv_mean'],
        'improvement_over_baseline': ((best_res['accuracy'] - baseline_acc) / baseline_acc) * 100,
        'features_used': len(numeric_features) + len(categorical_features),
        'advanced_features': True,
        'class_names': ['Low', 'Medium', 'High'],
        'thresholds': {'low_medium': q33, 'medium_high': q67}
    }
    joblib.dump(metadata, os.path.join(model_dir, 'roi_classifier_advanced_metadata.pkl'))
    print(f"✓ Metadata saved")

print("\n" + "=" * 80)
print("SUMMARY OF IMPROVEMENTS")
print("=" * 80)
print("\nTechniques Applied:")
print("  1. Advanced feature engineering (14 new features)")
print("  2. Polynomial features (squared terms for key variables)")
print("  3. Interaction features (revenue × time, investment × speed)")
print("  4. Binned features (investment/revenue quartiles)")
print("  5. Domain-specific features (benefit flags)")
print("  6. Optimized hyperparameters (deeper trees, more estimators)")
print("  7. Weighted voting ensemble (4 models)")
print("  8. Stacking ensemble (4 base + meta learner)")

if best_res['accuracy'] > 0.60:
    print(f"\n✓ EXCELLENT: Achieved {best_res['accuracy']*100:.1f}% accuracy (>60%)")
    print("  → Highly suitable for production use")
elif best_res['accuracy'] > 0.55:
    print(f"\n✓ VERY GOOD: Achieved {best_res['accuracy']*100:.1f}% accuracy (>55%)")
    print("  → Suitable for production use")
elif best_res['accuracy'] > baseline_acc:
    print(f"\n✓ IMPROVED: Achieved {best_res['accuracy']*100:.1f}% accuracy")
    improvement_pct = ((best_res['accuracy'] - baseline_acc) / baseline_acc) * 100
    print(f"  → {improvement_pct:.1f}% better than baseline")
else:
    print(f"\n⚠️  Limited improvement: {best_res['accuracy']*100:.1f}% accuracy")

print("\n" + "=" * 80)

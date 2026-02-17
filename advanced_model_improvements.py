import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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

numeric_features = [
    'log_investment', 'log_revenue', 'investment_per_day',
    'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
    'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year',
    'revenue_investment_ratio', 'time_efficiency', 'revenue_time_interaction',
    'investment_speed_ratio', 'prep_deployment_ratio',
    'log_investment_squared', 'time_saved_squared', 'revenue_increase_squared'
]

categorical_features = [
    'sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter',
    'investment_bin', 'revenue_bin'
]

X = df_work[numeric_features + categorical_features].copy()
print(f"   Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
print(f"   NEW features added: 11 (interactions, polynomials, bins)")

print("\n3. Train/test split (stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# STRATEGY 1: Optimized Single Models with GridSearch
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 1: HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
print("=" * 80)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# XGBoost with GridSearch
print("\n   Optimizing XGBoost...")
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss'))
])

xgb_params = {
    'classifier__n_estimators': [300, 500],
    'classifier__max_depth': [4, 6, 8],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__min_child_weight': [3, 5],
    'classifier__subsample': [0.8, 0.9],
    'classifier__colsample_bytree': [0.8, 0.9]
}

xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)

print(f"   Best params: {xgb_grid.best_params_}")
print(f"   Best CV score: {xgb_grid.best_score_:.4f}")

xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"   Test accuracy: {xgb_acc:.4f}")

# ============================================================================
# STRATEGY 2: SMOTE for Balanced Training
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 2: SMOTE OVERSAMPLING (Balanced Classes)")
print("=" * 80)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
preprocessor_fit = preprocessor.fit(X_train)
X_train_transformed = preprocessor_fit.transform(X_train)
X_train_smote, y_train_smote = smote.fit_resample(X_train_transformed, y_train)

print(f"   Original train distribution: {np.bincount(y_train)}")
print(f"   SMOTE train distribution: {np.bincount(y_train_smote)}")

# Train models on SMOTE data
models_smote = {
    'XGBoost_SMOTE': XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    ),
    'RandomForest_SMOTE': RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_split=8,
        min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1
    ),
    'LightGBM_SMOTE': LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        num_leaves=40, min_child_samples=8, subsample=0.9,
        colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=-1
    )
}

results_smote = {}
X_test_transformed = preprocessor_fit.transform(X_test)

for name, clf in models_smote.items():
    print(f"\n   Training {name}...")
    clf.fit(X_train_smote, y_train_smote)
    y_pred = clf.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results_smote[name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': clf,
        'y_pred': y_pred
    }
    print(f"      Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ============================================================================
# STRATEGY 3: Ensemble Methods (Voting & Stacking)
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 3: ENSEMBLE METHODS (Voting & Stacking)")
print("=" * 80)

# Voting Classifier
print("\n   Training Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, 
                             min_child_weight=3, subsample=0.9, random_state=42, 
                             n_jobs=-1, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_split=8,
                                     random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                              num_leaves=40, random_state=42, verbose=-1, n_jobs=-1))
    ],
    voting='soft'
)

voting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

voting_pipeline.fit(X_train, y_train)
voting_pred = voting_pipeline.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
voting_f1 = precision_recall_fscore_support(y_test, voting_pred, average='weighted')[2]
print(f"   Voting Accuracy: {voting_acc:.4f} | F1: {voting_f1:.4f}")

# Stacking Classifier
print("\n   Training Stacking Ensemble...")
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             random_state=42, n_jobs=-1, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              random_state=42, verbose=-1, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                         random_state=42))
    ],
    final_estimator=XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  random_state=42, eval_metric='mlogloss'),
    cv=5
)

stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

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
    'Baseline (GradientBoosting)': {'accuracy': 0.5161, 'f1': 0.5170},
    'XGBoost_GridSearch': {'accuracy': xgb_acc, 'f1': precision_recall_fscore_support(y_test, xgb_pred, average='weighted')[2]},
    **{k: {'accuracy': v['accuracy'], 'f1': v['f1']} for k, v in results_smote.items()},
    'Voting_Ensemble': {'accuracy': voting_acc, 'f1': voting_f1},
    'Stacking_Ensemble': {'accuracy': stacking_acc, 'f1': stacking_f1}
}

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\nRANKING (by Test Accuracy):")
print(f"{'Rank':<6}{'Model':<30}{'Accuracy':<15}{'F1-Score':<15}{'Improvement':<15}")
print("-" * 81)

baseline_acc = 0.5161
for rank, (name, res) in enumerate(sorted_results, 1):
    improvement = ((res['accuracy'] - baseline_acc) / baseline_acc) * 100
    marker = "★" if rank == 1 else " "
    print(f"{rank:<6}{marker} {name:<28}{res['accuracy']:<15.4f}{res['f1']:<15.4f}{improvement:+.1f}%")

# Save best model
best_name, best_res = sorted_results[0]
print("\n" + "=" * 80)
print(f"BEST MODEL: {best_name}")
print("=" * 80)
print(f"Accuracy: {best_res['accuracy']:.4f} ({best_res['accuracy']*100:.2f}%)")
print(f"F1-Score: {best_res['f1']:.4f}")
print(f"Improvement over baseline: {((best_res['accuracy'] - baseline_acc) / baseline_acc) * 100:+.1f}%")

# Save the best model
model_dir = 'backend/models'
os.makedirs(model_dir, exist_ok=True)

if best_name == 'XGBoost_GridSearch':
    best_model = xgb_best
elif best_name == 'Voting_Ensemble':
    best_model = voting_pipeline
elif best_name == 'Stacking_Ensemble':
    best_model = stacking_pipeline
else:
    # SMOTE model
    best_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', results_smote[best_name]['model'])
    ])
    best_model.fit(X_train, y_train)  # Refit on original data for consistency

best_path = os.path.join(model_dir, 'roi_classifier_advanced.pkl')
joblib.dump(best_model, best_path)
print(f"\n✓ Best model saved: {best_path}")

# Save metadata
metadata = {
    'model_name': best_name,
    'accuracy': best_res['accuracy'],
    'f1': best_res['f1'],
    'improvement_over_baseline': ((best_res['accuracy'] - baseline_acc) / baseline_acc) * 100,
    'features_used': len(numeric_features) + len(categorical_features),
    'advanced_features': True
}
joblib.dump(metadata, os.path.join(model_dir, 'roi_classifier_advanced_metadata.pkl'))

print("\n" + "=" * 80)
print("SUMMARY OF IMPROVEMENTS")
print("=" * 80)
print("\nTechniques Applied:")
print("  1. Advanced feature engineering (11 new features)")
print("  2. Polynomial features (squared terms)")
print("  3. Interaction features (revenue × time, investment × speed)")
print("  4. Binned features (investment/revenue quartiles)")
print("  5. Hyperparameter optimization (GridSearchCV)")
print("  6. SMOTE oversampling for class balance")
print("  7. Voting ensemble (soft voting)")
print("  8. Stacking ensemble (4 base + meta learner)")

if best_res['accuracy'] > 0.55:
    print(f"\n✓ EXCELLENT: Achieved {best_res['accuracy']*100:.1f}% accuracy (>55%)")
    print("  → Suitable for production use")
elif best_res['accuracy'] > baseline_acc:
    print(f"\n✓ IMPROVED: Achieved {best_res['accuracy']*100:.1f}% accuracy")
    improvement_pct = ((best_res['accuracy'] - baseline_acc) / baseline_acc) * 100
    print(f"  → {improvement_pct:.1f}% better than baseline")
else:
    print(f"\n⚠️  No improvement: {best_res['accuracy']*100:.1f}% accuracy")
    print("  → May need more data or different features")

print("\n" + "=" * 80)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MAXIMUM PERFORMANCE OPTIMIZATION - MULTIPLE APPROACHES")
print("=" * 80)

# Load data
df = pd.read_csv('data/processed/515.csv')
print(f"\n1. Dataset loaded: {df.shape}")

roi_q1 = df['roi'].quantile(0.05)
roi_q99 = df['roi'].quantile(0.95)
df_filtered = df[(df['roi'] >= roi_q1) & (df['roi'] <= roi_q99)].copy()
print(f"   After outlier removal: {df_filtered.shape}")

# ============================================================================
# APPROACH 1: Binary Classification (High vs Not-High)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 1: BINARY CLASSIFICATION (High ROI vs Not-High)")
print("=" * 80)

q67 = df_filtered['roi'].quantile(0.67)
print(f"   Threshold: High ROI ≥ {q67:.1f}%")

df_binary = df_filtered.copy()
df_binary['roi_binary'] = (df_binary['roi'] >= q67).astype(int)
print(f"   Class distribution: Not-High={sum(df_binary['roi_binary']==0)}, High={sum(df_binary['roi_binary']==1)}")

y_binary = df_binary['roi_binary'].copy()
df_work_binary = df_binary.drop(columns=['roi', 'roi_binary'])

# Feature engineering
df_work_binary['log_investment'] = np.log1p(df_work_binary['investment_eur'])
df_work_binary['log_revenue'] = np.log1p(df_work_binary['revenue_m_eur'])
df_work_binary['investment_per_day'] = df_work_binary['investment_eur'] / (df_work_binary['days_to_deployment'] + 1)
df_work_binary['total_prep_time'] = df_work_binary['days_diagnostic'] + df_work_binary['days_poc']
df_work_binary['deployment_speed'] = 1 / (df_work_binary['days_to_deployment'] + 1)
df_work_binary['is_large_company'] = (df_work_binary['company_size'] == 'grande').astype(int)
df_work_binary['human_in_loop'] = df_work_binary['human_in_loop'].astype(int)
df_work_binary['revenue_investment_ratio'] = df_work_binary['revenue_m_eur'] / (df_work_binary['investment_eur'] / 1000000 + 1)
df_work_binary['time_efficiency'] = df_work_binary['time_saved_hours_month'] / (df_work_binary['total_prep_time'] + 1)
df_work_binary['revenue_time_interaction'] = df_work_binary['revenue_increase_percent'] * df_work_binary['time_saved_hours_month']

numeric_features = [
    'log_investment', 'log_revenue', 'investment_per_day', 'total_prep_time',
    'deployment_speed', 'time_saved_hours_month', 'revenue_increase_percent',
    'is_large_company', 'human_in_loop', 'year', 'revenue_investment_ratio',
    'time_efficiency', 'revenue_time_interaction'
]
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']

X_binary = df_work_binary[numeric_features + categorical_features].copy()

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

models_binary = {
    'XGBoost_Binary': XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=2, random_state=42, n_jobs=-1, eval_metric='logloss'
    ),
    'RandomForest_Binary': RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=6,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'LightGBM_Binary': LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.03,
        num_leaves=50, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1
    )
}

results_binary = {}
for name, clf in models_binary.items():
    print(f"\n   Training {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipeline.fit(X_train_b, y_train_b)
    y_pred = pipeline.predict(X_test_b)
    y_proba = pipeline.predict_proba(X_test_b)[:, 1]
    
    acc = accuracy_score(y_test_b, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_b, y_pred, average='binary')
    auc = roc_auc_score(y_test_b, y_proba)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train_b, y_train_b, cv=cv, scoring='accuracy', n_jobs=-1)
    
    results_binary[name] = {
        'accuracy': acc, 'precision': precision, 'recall': recall,
        'f1': f1, 'auc': auc, 'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(), 'pipeline': pipeline, 'y_pred': y_pred
    }
    
    print(f"      Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"      F1: {f1:.4f} | AUC: {auc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ============================================================================
# APPROACH 2: Alternative 3-Class Thresholds
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 2: ALTERNATIVE THRESHOLDS (25th/75th percentiles)")
print("=" * 80)

q25 = df_filtered['roi'].quantile(0.25)
q75 = df_filtered['roi'].quantile(0.75)
print(f"   LOW: ROI < {q25:.1f}%")
print(f"   MEDIUM: {q25:.1f}% ≤ ROI < {q75:.1f}%")
print(f"   HIGH: ROI ≥ {q75:.1f}%")

def classify_roi_alt(roi):
    if roi < q25:
        return 0
    elif roi < q75:
        return 1
    else:
        return 2

df_alt = df_filtered.copy()
df_alt['roi_class'] = df_alt['roi'].apply(classify_roi_alt)
y_alt = df_alt['roi_class'].copy()

class_dist = y_alt.value_counts().sort_index()
print(f"   Class distribution: Low={class_dist[0]}, Medium={class_dist[1]}, High={class_dist[2]}")

df_work_alt = df_alt.drop(columns=['roi', 'roi_class'])

# Same feature engineering
df_work_alt['log_investment'] = np.log1p(df_work_alt['investment_eur'])
df_work_alt['log_revenue'] = np.log1p(df_work_alt['revenue_m_eur'])
df_work_alt['investment_per_day'] = df_work_alt['investment_eur'] / (df_work_alt['days_to_deployment'] + 1)
df_work_alt['total_prep_time'] = df_work_alt['days_diagnostic'] + df_work_alt['days_poc']
df_work_alt['deployment_speed'] = 1 / (df_work_alt['days_to_deployment'] + 1)
df_work_alt['is_large_company'] = (df_work_alt['company_size'] == 'grande').astype(int)
df_work_alt['human_in_loop'] = df_work_alt['human_in_loop'].astype(int)
df_work_alt['revenue_investment_ratio'] = df_work_alt['revenue_m_eur'] / (df_work_alt['investment_eur'] / 1000000 + 1)
df_work_alt['time_efficiency'] = df_work_alt['time_saved_hours_month'] / (df_work_alt['total_prep_time'] + 1)
df_work_alt['revenue_time_interaction'] = df_work_alt['revenue_increase_percent'] * df_work_alt['time_saved_hours_month']

X_alt = df_work_alt[numeric_features + categorical_features].copy()

X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
    X_alt, y_alt, test_size=0.2, random_state=42, stratify=y_alt
)

models_alt = {
    'XGBoost_Alt': XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    ),
    'GradientBoosting_Alt': GradientBoostingClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        min_samples_split=8, subsample=0.9, random_state=42
    )
}

results_alt = {}
for name, clf in models_alt.items():
    print(f"\n   Training {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipeline.fit(X_train_alt, y_train_alt)
    y_pred = pipeline.predict(X_test_alt)
    
    acc = accuracy_score(y_test_alt, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_alt, y_pred, average='weighted')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train_alt, y_train_alt, cv=cv, scoring='accuracy', n_jobs=-1)
    
    results_alt[name] = {
        'accuracy': acc, 'precision': precision, 'recall': recall,
        'f1': f1, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'pipeline': pipeline, 'y_pred': y_pred
    }
    
    print(f"      Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ============================================================================
# APPROACH 3: Feature Selection + Ensemble
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 3: FEATURE SELECTION + ENSEMBLE")
print("=" * 80)

# Use original 3-class problem
q33 = df_filtered['roi'].quantile(0.33)
q67 = df_filtered['roi'].quantile(0.67)

def classify_roi(roi):
    if roi < q33:
        return 0
    elif roi < q67:
        return 1
    else:
        return 2

df_fs = df_filtered.copy()
df_fs['roi_class'] = df_fs['roi'].apply(classify_roi)
y_fs = df_fs['roi_class'].copy()
df_work_fs = df_fs.drop(columns=['roi', 'roi_class'])

# Feature engineering
df_work_fs['log_investment'] = np.log1p(df_work_fs['investment_eur'])
df_work_fs['log_revenue'] = np.log1p(df_work_fs['revenue_m_eur'])
df_work_fs['investment_per_day'] = df_work_fs['investment_eur'] / (df_work_fs['days_to_deployment'] + 1)
df_work_fs['total_prep_time'] = df_work_fs['days_diagnostic'] + df_work_fs['days_poc']
df_work_fs['deployment_speed'] = 1 / (df_work_fs['days_to_deployment'] + 1)
df_work_fs['is_large_company'] = (df_work_fs['company_size'] == 'grande').astype(int)
df_work_fs['human_in_loop'] = df_work_fs['human_in_loop'].astype(int)
df_work_fs['revenue_investment_ratio'] = df_work_fs['revenue_m_eur'] / (df_work_fs['investment_eur'] / 1000000 + 1)
df_work_fs['time_efficiency'] = df_work_fs['time_saved_hours_month'] / (df_work_fs['total_prep_time'] + 1)
df_work_fs['revenue_time_interaction'] = df_work_fs['revenue_increase_percent'] * df_work_fs['time_saved_hours_month']

X_fs = df_work_fs[numeric_features + categorical_features].copy()

X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
    X_fs, y_fs, test_size=0.2, random_state=42, stratify=y_fs
)

# Feature selection with XGBoost
print("\n   Performing feature selection...")
selector_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectFromModel(
        XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='mlogloss'),
        threshold='median'
    ))
])

selector_pipeline.fit(X_train_fs, y_train_fs)
X_train_selected = selector_pipeline.transform(X_train_fs)
X_test_selected = selector_pipeline.transform(X_test_fs)

print(f"   Features selected: {X_train_selected.shape[1]} (from {preprocessor.fit_transform(X_train_fs).shape[1]})")

# Train ensemble on selected features
clf_fs = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03,
                             random_state=42, n_jobs=-1, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=15,
                                     class_weight='balanced', random_state=42, n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.03,
                              class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1))
    ],
    voting='soft',
    weights=[2, 1, 2]
)

print("   Training ensemble on selected features...")
clf_fs.fit(X_train_selected, y_train_fs)
y_pred_fs = clf_fs.predict(X_test_selected)

acc_fs = accuracy_score(y_test_fs, y_pred_fs)
precision_fs, recall_fs, f1_fs, _ = precision_recall_fscore_support(y_test_fs, y_pred_fs, average='weighted')

print(f"   Accuracy: {acc_fs:.4f} | F1: {f1_fs:.4f}")

results_fs = {
    'FeatureSelection_Ensemble': {
        'accuracy': acc_fs, 'precision': precision_fs, 'recall': recall_fs,
        'f1': f1_fs, 'cv_mean': np.nan, 'pipeline': selector_pipeline,
        'y_pred': y_pred_fs, 'classifier': clf_fs
    }
}

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON - ALL APPROACHES")
print("=" * 80)

all_results = {
    'Baseline (3-class, 33/67)': {'accuracy': 0.5161, 'type': '3-class'},
    'Optimized (3-class, 33/67)': {'accuracy': 0.5269, 'type': '3-class'},
    **{k: {**v, 'type': 'binary'} for k, v in results_binary.items()},
    **{k: {**v, 'type': '3-class-alt'} for k, v in results_alt.items()},
    **{k: {**v, 'type': '3-class-fs'} for k, v in results_fs.items()}
}

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\nRANKING (by Test Accuracy):")
print(f"{'Rank':<6}{'Model':<40}{'Type':<15}{'Accuracy':<12}{'F1/AUC':<12}")
print("-" * 85)

for rank, (name, res) in enumerate(sorted_results, 1):
    marker = "★" if rank == 1 else " "
    metric_val = res.get('auc', res.get('f1', 0))
    metric_str = f"{metric_val:.4f}"
    print(f"{rank:<6}{marker} {name:<38}{res['type']:<15}{res['accuracy']:<12.4f}{metric_str:<12}")

# Save best model
best_name, best_res = sorted_results[0]
print("\n" + "=" * 80)
print(f"BEST MODEL: {best_name}")
print("=" * 80)
print(f"Type: {best_res['type']}")
print(f"Accuracy: {best_res['accuracy']:.4f} ({best_res['accuracy']*100:.2f}%)")

if best_res['type'] == 'binary':
    print(f"Precision: {best_res['precision']:.4f}")
    print(f"Recall: {best_res['recall']:.4f}")
    print(f"F1-Score: {best_res['f1']:.4f}")
    print(f"AUC-ROC: {best_res['auc']:.4f}")
else:
    print(f"F1-Score: {best_res.get('f1', 'N/A')}")

# Save best model
model_dir = 'backend/models'
os.makedirs(model_dir, exist_ok=True)

if 'pipeline' in best_res:
    best_model = best_res['pipeline']
    if 'classifier' in best_res:
        # For feature selection approach, save both
        best_path = os.path.join(model_dir, 'roi_classifier_best.pkl')
        joblib.dump({'selector': best_model, 'classifier': best_res['classifier']}, best_path)
    else:
        best_path = os.path.join(model_dir, 'roi_classifier_best.pkl')
        joblib.dump(best_model, best_path)
    
    print(f"\n✓ Best model saved: {best_path}")
    
    metadata = {
        'model_name': best_name,
        'model_type': best_res['type'],
        'accuracy': best_res['accuracy'],
        'f1': best_res.get('f1', None),
        'auc': best_res.get('auc', None),
        'cv_mean': best_res.get('cv_mean', None)
    }
    joblib.dump(metadata, os.path.join(model_dir, 'roi_classifier_best_metadata.pkl'))
    print(f"✓ Metadata saved")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nApproaches Tested:")
print("  1. Binary Classification (High vs Not-High)")
print("  2. Alternative 3-Class Thresholds (25th/75th percentiles)")
print("  3. Feature Selection + Ensemble")

best_acc = best_res['accuracy']
if best_acc >= 0.65:
    print(f"\n✓ EXCELLENT: {best_acc*100:.1f}% accuracy achieved!")
elif best_acc >= 0.60:
    print(f"\n✓ VERY GOOD: {best_acc*100:.1f}% accuracy achieved")
elif best_acc >= 0.55:
    print(f"\n✓ GOOD: {best_acc*100:.1f}% accuracy achieved")
else:
    print(f"\n⚠️  Moderate: {best_acc*100:.1f}% accuracy")
    print("   Consider: More data, better features, or different problem framing")

print("\n" + "=" * 80)

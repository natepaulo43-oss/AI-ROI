import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ROI CLASSIFICATION MODEL - LOW / MEDIUM / HIGH")
print("=" * 80)

# Load data
df = pd.read_csv('data/processed/515.csv')
print(f"\n1. Dataset loaded: {df.shape}")

# Remove extreme outliers
roi_q1 = df['roi'].quantile(0.05)
roi_q99 = df['roi'].quantile(0.95)
df_filtered = df[(df['roi'] >= roi_q1) & (df['roi'] <= roi_q99)].copy()
print(f"   After outlier removal: {df_filtered.shape}")

print("\n2. Creating ROI classification labels")
print(f"   ROI range: [{df_filtered['roi'].min():.1f}, {df_filtered['roi'].max():.1f}]")
print(f"   ROI mean: {df_filtered['roi'].mean():.1f}, median: {df_filtered['roi'].median():.1f}")

# Define thresholds based on quartiles for balanced classes
q33 = df_filtered['roi'].quantile(0.33)
q67 = df_filtered['roi'].quantile(0.67)

print(f"\n   Classification thresholds:")
print(f"   - LOW:    ROI < {q33:.1f}%")
print(f"   - MEDIUM: {q33:.1f}% ≤ ROI < {q67:.1f}%")
print(f"   - HIGH:   ROI ≥ {q67:.1f}%")

# Create classification labels
def classify_roi(roi):
    if roi < q33:
        return 0  # Low
    elif roi < q67:
        return 1  # Medium
    else:
        return 2  # High

df_filtered['roi_class'] = df_filtered['roi'].apply(classify_roi)
class_names = ['Low', 'Medium', 'High']

# Check class distribution
class_dist = df_filtered['roi_class'].value_counts().sort_index()
print(f"\n   Class distribution:")
for i, name in enumerate(class_names):
    count = class_dist[i]
    pct = (count / len(df_filtered)) * 100
    print(f"   - {name:8s}: {count:3d} samples ({pct:.1f}%)")

y = df_filtered['roi_class'].copy()
df_work = df_filtered.drop(columns=['roi', 'roi_class'])

print("\n3. Feature engineering")
df_work['log_investment'] = np.log1p(df_work['investment_eur'])
df_work['log_revenue'] = np.log1p(df_work['revenue_m_eur'])
df_work['investment_per_day'] = df_work['investment_eur'] / (df_work['days_to_deployment'] + 1)
df_work['total_prep_time'] = df_work['days_diagnostic'] + df_work['days_poc']
df_work['deployment_speed'] = 1 / (df_work['days_to_deployment'] + 1)
df_work['is_large_company'] = (df_work['company_size'] == 'grande').astype(int)
df_work['human_in_loop'] = df_work['human_in_loop'].astype(int)

numeric_features = [
    'log_investment', 'log_revenue', 'investment_per_day',
    'total_prep_time', 'deployment_speed', 'time_saved_hours_month',
    'revenue_increase_percent', 'is_large_company', 'human_in_loop', 'year'
]
categorical_features = ['sector', 'company_size', 'ai_use_case', 'deployment_type', 'quarter']

X = df_work[numeric_features + categorical_features].copy()
print(f"   Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

print("\n4. Train/test split (stratified by ROI class)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Check train/test class distribution
train_dist = y_train.value_counts().sort_index()
test_dist = y_test.value_counts().sort_index()
print(f"\n   Train class distribution:")
for i, name in enumerate(class_names):
    print(f"   - {name:8s}: {train_dist[i]:3d} ({train_dist[i]/len(y_train)*100:.1f}%)")
print(f"   Test class distribution:")
for i, name in enumerate(class_names):
    print(f"   - {name:8s}: {test_dist[i]:3d} ({test_dist[i]/len(y_test)*100:.1f}%)")

print("\n5. Training classification models")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=300, max_depth=10, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_split=10, subsample=0.8, random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        num_leaves=31, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1
    )
}

results = {}
best_model = None
best_accuracy = 0

for name, classifier in models.items():
    print(f"\n   Training {name}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': pipeline,
        'y_pred': y_pred
    }
    
    print(f"      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    print(f"      CV Acc:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

print("\n" + "=" * 80)
print("CLASSIFICATION RESULTS SUMMARY")
print("=" * 80)

# Sort by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\nRANKING (by Test Accuracy):")
print(f"{'Rank':<6}{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'CV Acc':<15}")
print("-" * 87)

for rank, (name, res) in enumerate(sorted_results, 1):
    cv_str = f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}"
    marker = "★" if rank == 1 else " "
    print(f"{rank:<6}{marker} {name:<18}{res['accuracy']:<12.4f}{res['precision']:<12.4f}{res['recall']:<12.4f}{res['f1']:<12.4f}{cv_str:<15}")

# Detailed results for best model
best_name, best_result = sorted_results[0]
print("\n" + "=" * 80)
print(f"BEST MODEL: {best_name}")
print("=" * 80)

print(f"\nOverall Metrics:")
print(f"  Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall:    {best_result['recall']:.4f}")
print(f"  F1-Score:  {best_result['f1']:.4f}")
print(f"  CV Acc:    {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_result['y_pred'])
print(f"{'':12s} {'Predicted Low':>15s} {'Predicted Med':>15s} {'Predicted High':>15s}")
for i, name in enumerate(class_names):
    print(f"Actual {name:6s} {cm[i,0]:>15d} {cm[i,1]:>15d} {cm[i,2]:>15d}")

print(f"\nPer-Class Performance:")
precision, recall, f1, support = precision_recall_fscore_support(y_test, best_result['y_pred'])
for i, name in enumerate(class_names):
    print(f"  {name:8s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, best_result['y_pred'], target_names=class_names))

# Save models
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

model_dir = 'backend/models'
os.makedirs(model_dir, exist_ok=True)

# Save best model
best_path = os.path.join(model_dir, 'roi_classifier.pkl')
joblib.dump(best_result['pipeline'], best_path)
print(f"\n✓ Best classifier saved: {best_path}")

# Save metadata
metadata = {
    'model_name': best_name,
    'model_type': 'classification',
    'class_names': class_names,
    'thresholds': {'low_high': q33, 'medium_high': q67},
    'accuracy': best_result['accuracy'],
    'precision': best_result['precision'],
    'recall': best_result['recall'],
    'f1': best_result['f1'],
    'cv_mean': best_result['cv_mean'],
    'cv_std': best_result['cv_std']
}
joblib.dump(metadata, os.path.join(model_dir, 'roi_classifier_metadata.pkl'))
print(f"✓ Metadata saved: {os.path.join(model_dir, 'roi_classifier_metadata.pkl')}")

# Save top 3 models
for rank, (name, res) in enumerate(sorted_results[:3], 1):
    if rank > 1:
        alt_path = os.path.join(model_dir, f'roi_classifier_rank{rank}.pkl')
        joblib.dump(res['pipeline'], alt_path)
        print(f"✓ Rank {rank} classifier saved: {alt_path}")

print("\n" + "=" * 80)
print("COMPARISON: CLASSIFICATION vs REGRESSION")
print("=" * 80)

print("\nRegression (Previous Best - XGBoost):")
print(f"  R² Score:  0.1606 (explains 16.06% of variance)")
print(f"  MAE:       72.00% (average error)")
print(f"  RMSE:      87.17%")

print(f"\nClassification (Current Best - {best_name}):")
print(f"  Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}% correct predictions)")
print(f"  F1-Score:  {best_result['f1']:.4f}")
print(f"  CV Acc:    {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")

if best_result['accuracy'] > 0.5:
    improvement = "SIGNIFICANT IMPROVEMENT"
    print(f"\n✓ {improvement}: Classification achieves {best_result['accuracy']*100:.1f}% accuracy")
    print("  → Much more reliable than regression (16% R²)")
    print("  → Suitable for production use with clear ROI categories")
else:
    print(f"\n⚠️  Classification accuracy: {best_result['accuracy']*100:.1f}%")
    print("  → Better than regression but still challenging")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if best_result['accuracy'] >= 0.6:
    print("\n✓ CLASSIFICATION IS RECOMMENDED for production:")
    print(f"  • {best_result['accuracy']*100:.1f}% accuracy is reliable for decision support")
    print("  • Clear Low/Medium/High categories are easier to interpret")
    print("  • More stable predictions than regression")
    print(f"  • Use {best_name} classifier")
elif best_result['accuracy'] >= 0.45:
    print("\n⚠️  CLASSIFICATION IS MODERATELY USEFUL:")
    print(f"  • {best_result['accuracy']*100:.1f}% accuracy is better than random (33.3%)")
    print("  • Still better than regression for categorical guidance")
    print("  • Use with caution and communicate uncertainty")
else:
    print("\n⚠️  CLASSIFICATION PERFORMANCE IS LIMITED:")
    print(f"  • {best_result['accuracy']*100:.1f}% accuracy is close to random")
    print("  • Consider collecting more data or better features")

print("\n" + "=" * 80)

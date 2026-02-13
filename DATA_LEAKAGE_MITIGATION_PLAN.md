# Data Leakage & Risk Mitigation Plan

**Date:** 2026-02-12
**Status:** 🔴 MEDIUM RISK - Action Required
**Audit Result:** Issues found requiring immediate attention

---

## 🚨 **CRITICAL ISSUES FOUND**

### 1. Timeline Data Inconsistency (24 records)
**Issue:** 24 records where `days_diagnostic + days_poc > days_to_deployment`

**Impact:** Logically impossible - preparation cannot exceed total time

**Examples:**
```
Record 220: diagnostic(87) + poc(77) = 164 > deployment(81)
Record 224: diagnostic(17) + poc(92) = 109 > deployment(82)
Record 227: diagnostic(14) + poc(156) = 170 > deployment(145)
```

**Root Cause:** Synthetic data generation logic error

**Fix Required:** ✅ Immediate - Clean or remove these records

---

### 2. Data Leakage in Practical Model
**Issue:** Model uses outcome variables (`time_saved`, `revenue_increase`) that are only known AFTER deployment

**Impact:** Cannot make true pre-deployment predictions

**Features with Leakage:**
- ❌ `time_saved_hours_month` (r=0.44 with ROI)
- ❌ `revenue_increase_percent` (r=0.14 with ROI)
- ❌ `has_time_savings` (derived from above)
- ❌ `has_revenue_increase` (derived from above)

**Mitigation:** ✅ Use Conservative model for pre-deployment predictions

---

### 3. Random Train/Test Split
**Issue:** Using random 80/20 split instead of temporal split

**Impact:** May not generalize to future time periods

**Current Split:**
- Random: Records from 2022-2025 mixed in both train and test
- **Should be:** Train on 2022-2024, test on 2025

**Fix Required:** ⚠️ Implement temporal validation

---

## 📋 **DETAILED RISK ASSESSMENT**

| Risk Category | Severity | Status | Model Affected |
|---------------|----------|---------|----------------|
| Timeline Inconsistency | 🔴 HIGH | Needs Fix | Both |
| Outcome Variable Leakage | 🟡 MEDIUM | Acknowledged | Practical Only |
| Random Split | 🟡 MEDIUM | Sub-optimal | Both |
| Duplicate Records | 🟢 LOW | Clean | N/A |
| Perfect Leakage | 🟢 LOW | None Found | N/A |

---

## ✅ **IMMEDIATE ACTIONS (Do Now)**

### Action 1: Fix Timeline Inconsistencies

```python
import pandas as pd

# Load data
df = pd.read_csv('data/processed/ai_roi_training_dataset_enhanced.csv')

# Identify problematic records
inconsistent = df[
    (df['days_diagnostic'] + df['days_poc'] > df['days_to_deployment'])
]

print(f"Found {len(inconsistent)} inconsistent records")

# Option A: Fix the timeline (recommended)
# Make total_deployment = diagnostic + poc + deployment_phase
df.loc[inconsistent.index, 'days_to_deployment'] = (
    df.loc[inconsistent.index, 'days_diagnostic'] +
    df.loc[inconsistent.index, 'days_poc'] +
    30  # Add 30 days for actual deployment phase
)

# Option B: Remove inconsistent records (if too many)
# df = df.drop(inconsistent.index)

# Save cleaned data
df.to_csv('data/processed/ai_roi_training_dataset_cleaned.csv', index=False)
print("Cleaned dataset saved!")
```

### Action 2: Enforce Model Usage Guidelines

**Create clear documentation:**

```
PREDICTION GUIDELINES:

Pre-Deployment (No ROI data yet):
✅ Use: roi_model_conservative.pkl
❌ Do NOT use: roi_model.pkl (Practical)
Accuracy: Low (R²=-0.02, MAE=89%)
Purpose: Rough estimate only

Mid-Deployment (1-3 months in):
✅ Use: roi_model.pkl (Practical)
Requirements: Must have time_saved OR revenue_increase data
Accuracy: Medium (R²=0.18, MAE=76%)
Purpose: Adjust predictions based on early signals

Post-Deployment (6+ months):
✅ Use: Actual measured ROI
Purpose: Update training data for future
```

---

## 🔧 **MEDIUM-TERM FIXES (This Week)**

### Fix 1: Implement Temporal Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Sort by time
df_sorted = df.sort_values(['year', 'quarter'])

# Time-based split: Train on past, test on future
split_date = '2025-Q1'
train = df_sorted[df_sorted['year'] < 2025]
test = df_sorted[df_sorted['year'] == 2025]

print(f"Train: {len(train)} records (2022-2024)")
print(f"Test: {len(test)} records (2025)")

# Alternative: Rolling window validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(df_sorted):
    # Train and evaluate
    pass
```

### Fix 2: Add Validation Checks

```python
def validate_prediction_inputs(data):
    """Validate inputs before making predictions"""

    # Check 1: Timeline consistency
    if 'days_diagnostic' in data and 'days_poc' in data:
        total_prep = data['days_diagnostic'] + data['days_poc']
        if total_prep > data['days_to_deployment']:
            raise ValueError(
                f"Timeline inconsistent: prep({total_prep}) > "
                f"deployment({data['days_to_deployment']})"
            )

    # Check 2: Model selection based on available data
    if 'time_saved_hours_month' in data or 'revenue_increase_percent' in data:
        recommended_model = "practical"
    else:
        recommended_model = "conservative"
        print("[WARNING] No outcome data available.")
        print("          Using Conservative model (low accuracy)")

    # Check 3: Value ranges
    assert data['investment_eur'] > 0, "Investment must be positive"
    assert data['revenue_m_eur'] > 0, "Revenue must be positive"
    assert data['days_to_deployment'] >= 30, "Deployment too fast (< 30 days)"

    return recommended_model
```

### Fix 3: Implement Drift Monitoring

```python
def monitor_model_drift():
    """Monitor if model performance degrades over time"""

    # Load recent predictions
    recent_predictions = pd.read_csv('predictions_log.csv')

    # Compare predicted vs actual ROI
    if 'actual_roi' in recent_predictions.columns:
        mae = abs(recent_predictions['predicted_roi'] -
                 recent_predictions['actual_roi']).mean()

        if mae > 100:  # Error > 100%
            print("[ALERT] Model drift detected!")
            print(f"        Current MAE: {mae:.1f}%")
            print("        Consider retraining...")
        else:
            print(f"[OK] Model performing well (MAE: {mae:.1f}%)")
```

---

## 🎯 **LONG-TERM IMPROVEMENTS (This Month)**

### 1. Ensemble Approach

Combine multiple models to reduce risk:

```python
def ensemble_prediction(X, mode='pre-deployment'):
    """Ensemble prediction to reduce variance and risk"""

    if mode == 'pre-deployment':
        # Conservative + Industry Benchmark
        conservative_pred = conservative_model.predict(X)[0]
        industry_avg = 89.5  # From research
        ensemble = 0.7 * conservative_pred + 0.3 * industry_avg

    elif mode == 'mid-deployment':
        # Practical + Conservative
        practical_pred = practical_model.predict(X)[0]
        conservative_pred = conservative_model.predict(X)[0]
        ensemble = 0.6 * practical_pred + 0.4 * conservative_pred

    return ensemble
```

### 2. Uncertainty Quantification

Add prediction intervals:

```python
from sklearn.ensemble import GradientBoostingRegressor

# Train with quantile loss for prediction intervals
model_lower = GradientBoostingRegressor(
    loss='quantile', alpha=0.1  # 10th percentile
)
model_upper = GradientBoostingRegressor(
    loss='quantile', alpha=0.9  # 90th percentile
)

# Predictions with uncertainty
pred = model.predict(X)[0]
lower = model_lower.predict(X)[0]
upper = model_upper.predict(X)[0]

print(f"Predicted ROI: {pred:.1f}%")
print(f"80% Confidence: [{lower:.1f}%, {upper:.1f}%]")
```

### 3. Feature Importance Validation

Ensure features make business sense:

```python
# Get feature importance
importances = model.feature_importances_
feature_names = model.feature_names_in_

# Flag suspicious features
for feat, imp in zip(feature_names, importances):
    if imp > 0.3:  # Single feature explains >30%
        print(f"[WARNING] {feat} has {imp:.1%} importance")
        print("          Verify this makes business sense!")
```

---

## 📊 **VALIDATION CHECKLIST**

Before deploying to production:

- [ ] **Data Quality**
  - [ ] No timeline inconsistencies
  - [ ] No duplicate records
  - [ ] All features within valid ranges
  - [ ] Categorical values are valid

- [ ] **Model Selection**
  - [ ] Conservative model for pre-deployment
  - [ ] Practical model only with outcome data
  - [ ] Clear documentation for users
  - [ ] Error handling for wrong model usage

- [ ] **Validation**
  - [ ] Temporal validation (not random split)
  - [ ] Test on 2025 data separately
  - [ ] Cross-validation R² documented
  - [ ] Prediction intervals calculated

- [ ] **Monitoring**
  - [ ] Logging all predictions
  - [ ] Tracking actual vs predicted
  - [ ] Drift detection in place
  - [ ] Alert system for high errors

- [ ] **Documentation**
  - [ ] Model limitations documented
  - [ ] Leakage risks acknowledged
  - [ ] Usage guidelines clear
  - [ ] Uncertainty quantified

---

## 🔍 **MODEL COMPARISON: CONSERVATIVE vs PRACTICAL**

| Aspect | Conservative Model | Practical Model |
|--------|-------------------|-----------------|
| **Data Leakage** | ✅ None | ❌ Yes (outcome variables) |
| **Pre-Deployment Use** | ✅ Yes | ❌ No |
| **Mid-Deployment Use** | ⚠️ Suboptimal | ✅ Yes |
| **R² Score** | -0.015 (poor) | 0.185 (better) |
| **MAE** | 88.7% | 75.6% |
| **Uncertainty** | Very High | High |
| **Production Ready** | ⚠️ With caveats | ⚠️ With caveats |

---

## 💡 **RECOMMENDED DEPLOYMENT STRATEGY**

### Phase 1: Fix Data (Immediate)
1. ✅ Run timeline consistency fix
2. ✅ Remove or fix 24 problematic records
3. ✅ Retrain both models with clean data
4. ✅ Validate on 2025 data separately

### Phase 2: Add Safeguards (This Week)
1. ⏳ Implement input validation
2. ⏳ Add model selection logic
3. ⏳ Create prediction intervals
4. ⏳ Set up monitoring dashboard

### Phase 3: Improve (This Month)
1. 🔲 Implement ensemble approach
2. 🔲 Add temporal cross-validation
3. 🔲 Build drift detection
4. 🔲 Create user documentation

### Phase 4: Validate (Before Production)
1. 🔲 A/B test predictions vs actual
2. 🔲 Compare to industry benchmarks
3. 🔲 Get stakeholder sign-off
4. 🔲 Deploy with monitoring

---

## 📞 **CONTACT & ESCALATION**

**Data Quality Issues:** Clean data immediately (script provided above)

**Model Performance Issues:** Consider ensemble or collect more data

**Production Deployment:** Implement all Phase 1 + Phase 2 fixes first

---

## 📈 **SUCCESS METRICS**

Track these after implementing fixes:

| Metric | Before Fixes | Target After Fixes |
|--------|-------------|-------------------|
| Timeline inconsistencies | 24 | 0 |
| R² (Conservative) | -0.015 | >0.05 |
| R² (Practical) | 0.185 | >0.20 |
| MAE (Conservative) | 88.7% | <85% |
| MAE (Practical) | 75.6% | <70% |
| Prediction errors > 100% | Unknown | <30% |

---

## ✅ **CONCLUSION**

**Current Status:** Models trained but have data quality and leakage issues

**Risk Level:** MEDIUM - Can be used with caution and fixes

**Required Actions:**
1. Fix timeline inconsistencies (24 records)
2. Document model limitations clearly
3. Implement proper model selection logic
4. Add validation and monitoring

**Timeline:** All fixes can be completed in 1-2 weeks

**Sign-off Required:** Data quality fixes before production deployment

---

**Last Updated:** 2026-02-12
**Next Review:** 2026-02-19 (after implementing Phase 1 & 2 fixes)

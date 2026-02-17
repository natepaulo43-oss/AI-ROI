# Training Results - New Dataset (ai_roi_full_combined_cleaned.csv)

## Executive Summary

**Date:** February 16, 2026  
**Dataset:** `data/processed/ai_roi_full_combined_cleaned.csv`  
**Total Samples:** 514 (462 after outlier removal)  
**Best Model:** RandomForest  
**Best Test R²:** 0.0703 (7.03%)  
**Status:** ⚠️ **POOR PERFORMANCE** - Model has limited predictive power

---

## Dataset Characteristics

### Size & Distribution
- **Total rows:** 514
- **Rows after outlier filtering (5th-95th percentile):** 462
- **Outliers removed:** 52 (10.1%)

### ROI Statistics (Full Dataset)
- **Range:** -30.0% to 3,750.0%
- **Mean:** 122.9%
- **Median:** 84.8%
- **Std Dev:** 215.9%
- **Negative ROI cases:** 78 (15.2%)
- **Extreme outliers (>500%):** 11 (2.1%)

### Data Quality Issues Fixed
- **Timeline inconsistencies:** 24 rows corrected
  - Issue: `days_diagnostic + days_poc > days_to_deployment`
  - Resolution: Adjusted `days_to_deployment` to be consistent

---

## Model Performance Comparison

### 1. RandomForest ★ BEST
- **Test R²:** 0.0703
- **Test MAE:** 67.41%
- **Test RMSE:** 83.49%
- **CV R² (5-fold):** 0.1577 ± 0.0447

### 2. GradientBoosting
- **Test R²:** -0.0428
- **Test MAE:** 68.19%
- **Test RMSE:** 88.42%
- **CV R² (5-fold):** 0.0804 ± 0.0787

### 3. LightGBM
- **Test R²:** 0.0052
- **Test MAE:** 70.64%
- **Test RMSE:** 86.36%
- **CV R² (5-fold):** 0.0346 ± 0.0491

---

## Feature Engineering Applied

### Numeric Features (11)
1. `log_investment` - Log transform of investment amount
2. `log_revenue` - Log transform of company revenue
3. `investment_ratio` - Investment as % of revenue
4. `investment_per_day` - Investment efficiency metric
5. `total_prep_time` - Diagnostic + PoC days
6. `deployment_speed` - Inverse of deployment time
7. `time_saved_hours_month` - Operational efficiency gain
8. `revenue_increase_percent` - Revenue impact
9. `is_large_company` - Binary flag for large enterprises
10. `human_in_loop` - Binary flag for human oversight
11. `year` - Temporal feature

### Categorical Features (5)
1. `sector` - Industry sector
2. `company_size` - SME, ETI, or Large
3. `ai_use_case` - Type of AI application
4. `deployment_type` - Cloud, on-premise, or hybrid
5. `quarter` - Seasonal factor

---

## Key Findings

### ❌ Critical Issues

1. **Very Low Predictive Power**
   - Best R² of 7% means the model explains only 7% of ROI variance
   - 93% of ROI variation is unexplained by available features
   - Cross-validation shows slight improvement (15.8%) but still poor

2. **High Prediction Error**
   - MAE of 67.4% means average prediction is off by 67 percentage points
   - For a project with actual 100% ROI, model might predict 33% or 167%

3. **Negative R² on Some Models**
   - GradientBoosting achieved negative R², performing worse than predicting the mean
   - Indicates overfitting or fundamental mismatch between features and target

### 📊 Data Challenges

1. **Small Dataset**
   - Only 462 usable samples after outlier removal
   - Insufficient for complex pattern learning
   - High variance in cross-validation scores

2. **High ROI Variance**
   - Standard deviation (215.9%) is 1.75x the mean (122.9%)
   - Extreme outliers (up to 3,750% ROI)
   - 15% of projects have negative ROI

3. **Feature-Target Mismatch**
   - Available features (investment, company size, sector, etc.) have weak correlation with ROI
   - ROI appears highly dependent on execution quality and external factors not captured in data

---

## Models Saved

All trained models have been saved to `backend/models/`:

1. **`roi_model.pkl`** - Best model (RandomForest) - **USE THIS FOR PRODUCTION**
2. `roi_model_gradientboosting.pkl` - Alternative model
3. `roi_model_lightgbm.pkl` - Alternative model
4. `roi_model_conservative.pkl` - Pre-adoption only features (from earlier training)

---

## Recommendations

### 🔴 Immediate Actions

1. **Do NOT use for precise predictions**
   - Model has <10% explanatory power
   - Predictions will have very high uncertainty
   - Use only for directional guidance

2. **Consider Classification Instead**
   - Switch from regression to classification
   - Categories: Low ROI (<50%), Medium (50-150%), High (>150%)
   - Classification may be more reliable than precise percentage prediction

3. **Communicate Uncertainty**
   - Always show confidence intervals (±67% at minimum)
   - Emphasize that ROI depends heavily on execution
   - Present predictions as rough estimates, not forecasts

### 📈 Long-term Improvements

1. **Collect More Data**
   - Target: 2,000+ samples for reliable modeling
   - Focus on diverse sectors and company sizes
   - Include failed projects (negative ROI) for balanced learning

2. **Capture Additional Features**
   - Team experience/expertise level
   - Data quality and availability
   - Change management effectiveness
   - Stakeholder buy-in metrics
   - Technical complexity scores
   - Vendor/technology maturity

3. **Improve Data Quality**
   - Standardize ROI calculation methodology
   - Validate extreme values (>500% ROI)
   - Add data collection checkpoints during project lifecycle

4. **Alternative Modeling Approaches**
   - Bayesian models with uncertainty quantification
   - Ensemble methods with prediction intervals
   - Survival analysis for time-to-ROI
   - Causal inference methods

---

## Technical Details

### Training Configuration
- **Train/Test Split:** 80/20
- **Cross-Validation:** 5-fold
- **Outlier Handling:** Removed values outside 5th-95th percentile
- **Scaling:** RobustScaler for numeric features
- **Encoding:** One-hot encoding for categorical features

### RandomForest Hyperparameters
```python
n_estimators=200
max_depth=10
min_samples_split=10
min_samples_leaf=5
random_state=42
```

---

## Conclusion

The new dataset has been successfully used to train ROI prediction models, but **performance is poor** due to:
- Small dataset size (514 samples)
- High ROI variance and outliers
- Weak correlation between available features and ROI outcomes

**The model should be used with extreme caution** and only for rough directional guidance. Consider implementing a classification approach or collecting significantly more data before deploying for production use.

For production deployment, use the RandomForest model (`roi_model.pkl`) but always communicate the high uncertainty in predictions.

# AI ROI Prediction Model - Documentation

## 📊 Model Overview

**Model Type:** Practical ROI Predictor (with early deployment signals)  
**Algorithm:** Gradient Boosting Regressor  
**Performance:** R² = 0.42, MAE = 62.67%, RMSE = 84.16%  
**Training Date:** February 2026  
**Dataset Size:** 200 AI deployment projects

---

## 🎯 Model Purpose

Predicts ROI percentage for AI deployments using:
- Pre-adoption characteristics (sector, company size, investment, etc.)
- **Early deployment signals** (time savings, revenue increase in first weeks/months)

**Use Case:** Estimate ROI after initial deployment data becomes available (not pure pre-adoption prediction).

---

## 📥 Required Input Features

### **Numeric Features (17)**

| Feature | Description | Example | Source |
|---------|-------------|---------|--------|
| `year` | Deployment year | 2024 | Pre-adoption |
| `revenue_m_eur` | Company revenue (millions EUR) | 330.7 | Pre-adoption |
| `investment_eur` | AI investment (EUR) | 353519 | Pre-adoption |
| `days_diagnostic` | Diagnostic phase duration | 35 | Pre-adoption |
| `days_poc` | Proof-of-concept duration | 115 | Pre-adoption |
| `days_to_deployment` | Total deployment time | 360 | Pre-adoption |
| `human_in_loop` | Human oversight (0/1) | 1 | Pre-adoption |
| `time_saved_hours_month` | Time savings per month | 552 | **Early signal** ⚠️ |
| `revenue_increase_percent` | Revenue increase % | 0.0 | **Early signal** ⚠️ |

**Engineered features** (created automatically):
- `log_investment`, `log_revenue` (log transforms)
- `investment_ratio`, `investment_per_day` (efficiency ratios)
- `diagnostic_efficiency`, `poc_efficiency` (time ratios)
- `total_prep_time`, `deployment_speed` (time metrics)
- `size_investment_interaction` (interaction term)
- `is_large_company`, `is_hybrid_deployment` (binary flags)
- `has_revenue_increase`, `has_time_savings` (outcome flags)

### **Categorical Features (5)**

| Feature | Description | Values | Example |
|---------|-------------|--------|---------|
| `quarter` | Deployment quarter | q1, q2, q3, q4 | q1 |
| `sector` | Industry sector | manufacturing, finance, retail, etc. | manufacturing |
| `company_size` | Company size category | pme, eti, grande | grande |
| `ai_use_case` | AI application type | customer service bot, predictive analytics, etc. | customer service bot |
| `deployment_type` | Deployment approach | analytics, nlp, hybrid, automation, vision | analytics |

---

## 📤 Output

**Target Variable:** `roi` (ROI percentage)
- **Range:** -27.2% to 411.6%
- **Mean:** 132.25%
- **Std Dev:** 101.77%

---

## 🔑 Key Insights

### **Feature Importance (Top 10)**

1. **time_saved_hours_month** (26.64%) - Most important predictor
2. **has_time_savings** (15.51%) - Binary flag for time savings
3. **sector_finance** (9.97%) - Finance sector indicator
4. **investment_ratio** (7.73%) - Investment relative to revenue
5. **deployment_speed** (4.18%) - Speed of deployment
6. **log_investment** (3.46%) - Log-transformed investment
7. **investment_per_day** (3.31%) - Daily investment rate
8. **human_in_loop** (3.24%) - Human oversight flag
9. **diagnostic_efficiency** (3.20%) - Diagnostic phase efficiency
10. **total_prep_time** (2.69%) - Total preparation time

### **Critical Finding**

⚠️ **Pre-adoption features alone have near-zero predictive power (R² < 0.1)**

ROI is primarily determined by:
- Execution quality during deployment
- Early operational outcomes (time savings, revenue impact)
- Post-deployment factors not captured in pre-adoption data

---

## 🚀 Usage

### **Training the Model**

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Train model
python backend\train_roi_model.py
```

**Output:**
- `backend/models/roi_model.pkl` (trained pipeline)
- Training metrics printed to console

### **Making Predictions**

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('backend/models/roi_model.pkl')

# Prepare input data (must include all required features)
input_data = pd.DataFrame({
    'year': [2024],
    'quarter': ['q1'],
    'sector': ['manufacturing'],
    'company_size': ['grande'],
    'revenue_m_eur': [330.7],
    'ai_use_case': ['customer service bot'],
    'deployment_type': ['analytics'],
    'days_diagnostic': [35],
    'days_poc': [115],
    'days_to_deployment': [360],
    'investment_eur': [353519],
    'time_saved_hours_month': [0],  # Early signal
    'revenue_increase_percent': [0.0],  # Early signal
    'human_in_loop': [1]
})

# Predict ROI
predicted_roi = model.predict(input_data)
print(f"Predicted ROI: {predicted_roi[0]:.2f}%")
```

---

## ⚠️ Limitations

1. **Requires Early Deployment Data**
   - Cannot predict ROI purely from pre-adoption features
   - Needs `time_saved_hours_month` and `revenue_increase_percent`
   - Best used after 1-3 months of deployment

2. **Moderate Predictive Power**
   - R² = 0.42 means model explains 42% of variance
   - 58% of ROI variation is due to unmeasured factors
   - Average prediction error: ±62.67%

3. **Small Training Dataset**
   - Only 200 samples
   - May not generalize to very different contexts
   - Limited representation of edge cases

4. **Missing Important Features**
   - Team experience/expertise
   - Vendor/technology quality
   - Change management effectiveness
   - Organizational readiness

---

## 🔮 Future Improvements

### **Short-term (Current Dataset)**
- ✅ Implemented: Advanced feature engineering
- ✅ Implemented: Gradient Boosting algorithm
- ✅ Implemented: Proper preprocessing pipeline

### **Medium-term (Data Collection)**
Collect additional features:
- **Team factors:** AI expertise level, team size, training hours
- **Vendor factors:** Vendor reputation score, support quality
- **Process factors:** Change management score, stakeholder buy-in
- **Context factors:** Competitive pressure, regulatory environment

### **Long-term (Model Evolution)**
- Expand dataset to 500+ samples
- Develop separate models per sector/use case
- Implement ensemble methods (stacking multiple models)
- Add confidence intervals to predictions
- Create classification model for ROI categories

---

## 📈 Model Performance History

| Version | Date | Algorithm | R² | MAE | Notes |
|---------|------|-----------|-----|-----|-------|
| v1.0 | Feb 2026 | RandomForest (pre-only) | 0.0044 | 89.78% | Failed - no predictive power |
| v2.0 | Feb 2026 | GradientBoosting (practical) | **0.4218** | **62.67%** | Current - includes early signals |

---

## 🔬 Technical Details

### **Algorithm Configuration**

```python
GradientBoostingRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    min_samples_split=8,
    random_state=42
)
```

### **Preprocessing Pipeline**

1. **Numeric features:** StandardScaler normalization
2. **Categorical features:** One-hot encoding
3. **Feature engineering:** Automated in training script

### **Train/Test Split**

- Training: 160 samples (80%)
- Testing: 40 samples (20%)
- Random state: 42 (reproducible)

---

## 📞 Support

For questions or issues:
1. Check this documentation
2. Review training logs in console output
3. Verify input data format matches requirements
4. Ensure all required features are present

---

## 📝 Changelog

**v2.0 (Feb 2026)**
- Switched to Gradient Boosting algorithm
- Added 10+ engineered features
- Included early deployment signals
- Achieved R² = 0.42 (vs 0.004 in v1.0)

**v1.0 (Feb 2026)**
- Initial RandomForest model
- Pre-adoption features only
- Poor performance (R² = 0.004)
- Deprecated

---

**Last Updated:** February 10, 2026  
**Model File:** `backend/models/roi_model.pkl`  
**Training Script:** `backend/train_roi_model.py`

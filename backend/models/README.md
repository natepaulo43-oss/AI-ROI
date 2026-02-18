# AI ROI Models Directory

This directory contains the production machine learning models used by the backend API.

## Active Models

### 1. Binary Classifier: `roi_classifier_best.pkl`
**Purpose:** Predicts whether an AI project will achieve High ROI (≥145.5%) or Not-High ROI (<145.5%)

**Performance Metrics:**
- **Accuracy:** 76.70%
- **AUC-ROC:** 76.74%
- **Average Confidence:** 75.5%
- **Algorithm:** Gradient Boosting Classifier
- **Validated:** 5-fold cross-validation

**Usage:**
- Primary model for ROI category prediction
- Returns binary classification with confidence scores
- Used in `/predict` endpoint for classification

**Metadata File:** `roi_classifier_best_metadata.pkl`

---

### 2. Regression Model: `roi_model.pkl`
**Purpose:** Predicts continuous ROI percentage value

**Performance Metrics:**
- **R² Score:** 0.42
- **MAE:** ±62.67%
- **Algorithm:** Gradient Boosting Regressor

**Usage:**
- Provides continuous ROI predictions
- Generates confidence intervals (lower/upper bounds)
- Creates monthly forecast trajectories
- Used in `/predict` endpoint for ROI value estimation

---

## Model Loading

Both models are loaded together via `backend/app/model_loader.py`:

```python
from backend.app.model_loader import load_model

models = load_model()
classifier = models['classifier']  # Binary classifier
regression = models['regression']   # Continuous predictor
```

## Model Files

```
backend/models/
├── .gitkeep
├── roi_classifier_best.pkl          # Binary classifier (846 KB)
├── roi_classifier_best_metadata.pkl # Classifier metadata (261 B)
└── roi_model.pkl                     # Regression model (1.09 MB)
```

## Training Scripts

To retrain these models:

**Classifier:**
```bash
python backend/train_roi_classifier.py
```

**Regression:**
```bash
python backend/train_roi_model.py
```

## Performance Analysis

Detailed performance visualizations are available in:
- `backend/research_output/classifier_performance_analysis.png` - Classifier metrics
- `backend/research_output/best_model_residuals.png` - Regression residuals

## Notes

- Models are cached in memory after first load for performance
- Use `clear_model_cache()` to reload models after retraining
- Both models use the same feature engineering pipeline
- Models are excluded from git via `.gitignore` (must be trained locally)

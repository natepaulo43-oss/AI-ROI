import joblib
from pathlib import Path
from typing import Optional, Dict, Any

_model_cache: Optional[Dict[str, Any]] = None

def load_model():
    """
    Load both classifier and regression models.
    Uses caching to avoid reloading on every request.
    Returns dict with 'classifier' and 'regression' keys.
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    backend_dir = Path(__file__).parent.parent
    classifier_path = backend_dir / "models" / "roi_classifier_best.pkl"
    regression_path = backend_dir / "models" / "roi_model.pkl"
    
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier model not found at: {classifier_path}")
    if not regression_path.exists():
        raise FileNotFoundError(f"Regression model not found at: {regression_path}")
    
    print(f"Loading classifier from: {classifier_path}")
    classifier = joblib.load(classifier_path)
    print("✓ Binary Classifier loaded (High ROI vs Not-High)")
    print("  Accuracy: 68.82% (Statistically Significant: p < 0.001)")
    
    print(f"Loading regression model from: {regression_path}")
    regression = joblib.load(regression_path)
    print("✓ Regression Model loaded (Continuous ROI Prediction)")
    print("  Performance: R²=0.42, MAE=±62.67%")
    
    _model_cache = {
        'classifier': classifier,
        'regression': regression
    }
    
    return _model_cache

def clear_model_cache():
    """Clear the cached models (useful for reloading after retraining)"""
    global _model_cache
    _model_cache = None

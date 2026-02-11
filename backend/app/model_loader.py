import joblib
from pathlib import Path
from typing import Optional

_model_cache: Optional[object] = None

def load_model(model_path: str = None):
    """
    Load the trained ROI prediction model.
    Uses caching to avoid reloading on every request.
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if model_path is None:
        # Default path relative to backend directory
        backend_dir = Path(__file__).parent.parent
        model_path = backend_dir / "models" / "roi_model.pkl"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    _model_cache = joblib.load(model_path)
    print("Model loaded successfully!")
    
    return _model_cache

def clear_model_cache():
    """Clear the cached model (useful for reloading after retraining)"""
    global _model_cache
    _model_cache = None

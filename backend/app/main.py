from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionInput, PredictionOutput
from .model_loader import load_model
from .predict import make_prediction

app = FastAPI(
    title="AI ROI Prediction API",
    description="Predict ROI for AI deployments using machine learning",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
models = None

@app.on_event("startup")
async def startup_event():
    """Load the models when the API starts"""
    global models
    try:
        models = load_model()
        print("✅ Models loaded successfully on startup")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models on startup: {e}")
        print("Models will be loaded on first prediction request")

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "AI ROI Prediction API",
        "version": "2.0",
        "status": "running",
        "models_loaded": models is not None
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "classifier_loaded": models is not None and 'classifier' in models,
        "regression_loaded": models is not None and 'regression' in models,
        "model_version": "v2.0_complete"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_roi(input_data: PredictionInput):
    """
    Predict ROI for an AI deployment.
    
    Requires all deployment characteristics including:
    - Company info (sector, size, revenue)
    - AI deployment details (use case, type, timeline)
    - Investment amount
    - Optional: Early deployment signals (time savings, revenue increase)
    
    Returns complete prediction with:
    - Binary classification (High vs Not-High ROI)
    - Probability scores and confidence
    - Continuous ROI prediction with confidence intervals
    - 12-month forecast with ramp-up trajectory
    """
    global models
    
    # Load models if not already loaded
    if models is None:
        try:
            models = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load models: {str(e)}"
            )
    
    # Make prediction
    try:
        result = make_prediction(models, input_data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

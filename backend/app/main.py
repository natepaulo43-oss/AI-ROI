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

# Load model on startup
model = None

@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts"""
    global model
    try:
        model = load_model()
        print("✅ Model loaded successfully on startup")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first prediction request")

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "AI ROI Prediction API",
        "version": "2.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": "v2.0_practical"
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
    
    Returns predicted ROI percentage with confidence note.
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    # Make prediction
    try:
        result = make_prediction(model, input_data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

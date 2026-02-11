from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="ROI Prediction API")

model = None
MODEL_PATH = os.path.join('backend', 'models', 'roi_model.pkl')

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

class PredictionRequest(BaseModel):
    investment_eur: float
    revenue_m_eur: float
    human_in_loop: int
    days_to_deployment: int
    days_diagnostic: int
    days_poc: int
    sector: str
    company_size: str
    ai_use_case: str
    deployment_type: str

class PredictionResponse(BaseModel):
    predicted_roi_percent: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_roi(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        log_investment = np.log1p(request.investment_eur)
        log_revenue = np.log1p(request.revenue_m_eur)
        investment_ratio = request.investment_eur / request.revenue_m_eur
        
        feature_dict = {
            'log_investment': [log_investment],
            'log_revenue': [log_revenue],
            'investment_ratio': [investment_ratio],
            'human_in_loop': [int(request.human_in_loop)],
            'days_to_deployment': [request.days_to_deployment],
            'days_diagnostic': [request.days_diagnostic],
            'days_poc': [request.days_poc],
            'sector': [request.sector],
            'company_size': [request.company_size],
            'ai_use_case': [request.ai_use_case],
            'deployment_type': [request.deployment_type]
        }
        features = pd.DataFrame(feature_dict)
        
        prediction = model.predict(features)[0]
        
        return PredictionResponse(predicted_roi_percent=float(prediction))
    
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="revenue_m_eur cannot be zero")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "ROI Prediction API", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

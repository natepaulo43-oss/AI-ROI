from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="ROI Prediction API")

model = None
MODEL_PATH = os.path.join('backend', 'models', 'roi_classifier_best.pkl')

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Best Binary Classifier loaded successfully from {MODEL_PATH}")
        print(f"   Model Type: Binary Classification (High ROI vs Not-High)")
        print(f"   Accuracy: 68.82% (Statistically Significant: p < 0.001)")
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
    time_saved_hours_month: float = 0.0
    revenue_increase_percent: float = 0.0
    year: int = 2024
    quarter: str = 'q1'
    sector: str
    company_size: str
    ai_use_case: str
    deployment_type: str

class PredictionResponse(BaseModel):
    prediction: str  # "High" or "Not-High"
    probability_high: float  # Probability of High ROI (0-1)
    probability_not_high: float  # Probability of Not-High ROI (0-1)
    confidence: float  # Confidence score (0-1)
    threshold: float  # ROI threshold for "High" classification (145.5%)
    interpretation: str  # Human-readable interpretation

@app.post("/predict", response_model=PredictionResponse)
async def predict_roi(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Feature engineering (matching training pipeline)
        log_investment = np.log1p(request.investment_eur)
        log_revenue = np.log1p(request.revenue_m_eur)
        investment_per_day = request.investment_eur / (request.days_to_deployment + 1)
        total_prep_time = request.days_diagnostic + request.days_poc
        deployment_speed = 1 / (request.days_to_deployment + 1)
        is_large_company = 1 if request.company_size == 'grande' else 0
        revenue_investment_ratio = request.revenue_m_eur / (request.investment_eur / 1000000 + 1)
        time_efficiency = request.time_saved_hours_month / (total_prep_time + 1)
        revenue_time_interaction = request.revenue_increase_percent * request.time_saved_hours_month
        
        # Build feature dictionary
        feature_dict = {
            'log_investment': [log_investment],
            'log_revenue': [log_revenue],
            'investment_per_day': [investment_per_day],
            'total_prep_time': [total_prep_time],
            'deployment_speed': [deployment_speed],
            'time_saved_hours_month': [request.time_saved_hours_month],
            'revenue_increase_percent': [request.revenue_increase_percent],
            'is_large_company': [is_large_company],
            'human_in_loop': [int(request.human_in_loop)],
            'year': [request.year],
            'revenue_investment_ratio': [revenue_investment_ratio],
            'time_efficiency': [time_efficiency],
            'revenue_time_interaction': [revenue_time_interaction],
            'sector': [request.sector],
            'company_size': [request.company_size],
            'ai_use_case': [request.ai_use_case],
            'deployment_type': [request.deployment_type],
            'quarter': [request.quarter]
        }
        features = pd.DataFrame(feature_dict)
        
        # Get prediction and probabilities
        prediction_binary = model.predict(features)[0]  # 0 = Not-High, 1 = High
        probabilities = model.predict_proba(features)[0]  # [prob_not_high, prob_high]
        
        prob_not_high = float(probabilities[0])
        prob_high = float(probabilities[1])
        confidence = max(prob_high, prob_not_high)
        
        # Interpret prediction
        if prediction_binary == 1:
            prediction_label = "High"
            interpretation = f"High ROI Expected (≥145.5%). Confidence: {prob_high*100:.1f}%"
        else:
            prediction_label = "Not-High"
            interpretation = f"Not-High ROI Expected (<145.5%). Confidence: {prob_not_high*100:.1f}%"
        
        return PredictionResponse(
            prediction=prediction_label,
            probability_high=prob_high,
            probability_not_high=prob_not_high,
            confidence=confidence,
            threshold=145.5,
            interpretation=interpretation
        )
    
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

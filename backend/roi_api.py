from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="ROI Prediction API")

classifier_model = None
regression_model = None
CLASSIFIER_PATH = os.path.join('models', 'roi_classifier_best.pkl')
REGRESSION_PATH = os.path.join('models', 'roi_model.pkl')

@app.on_event("startup")
async def load_model():
    global classifier_model, regression_model
    try:
        classifier_model = joblib.load(CLASSIFIER_PATH)
        print(f"✓ Binary Classifier loaded from {CLASSIFIER_PATH}")
        print(f"   Model Type: Binary Classification (High ROI vs Not-High)")
        print(f"   Accuracy: 76.70% | AUC-ROC: 76.74% | Avg Confidence: 75.5%")
        
        regression_model = joblib.load(REGRESSION_PATH)
        print(f"✓ Regression Model loaded from {REGRESSION_PATH}")
        print(f"   Model Type: Continuous ROI Prediction")
        print(f"   Performance: R²=0.42, MAE=±62.67%")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
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
    predicted_roi: float  # Predicted ROI percentage
    roi_lower_bound: float  # Lower confidence interval (predicted_roi - MAE)
    roi_upper_bound: float  # Upper confidence interval (predicted_roi + MAE)
    forecast_months: list  # Monthly ROI forecast for visualization

@app.post("/predict", response_model=PredictionResponse)
async def predict_roi(request: PredictionRequest):
    try:
        if classifier_model is None or regression_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
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
        
        # Get classification prediction and probabilities
        prediction_binary = classifier_model.predict(features)[0]  # 0 = Not-High, 1 = High
        probabilities = classifier_model.predict_proba(features)[0]  # [prob_not_high, prob_high]
        
        # Get continuous ROI prediction from regression model
        predicted_roi = float(regression_model.predict(features)[0])
        
        # Calculate confidence intervals (using MAE of 62.67% from model documentation)
        mae = 62.67
        roi_lower = predicted_roi - mae
        roi_upper = predicted_roi + mae
        
        # Generate monthly forecast based on deployment timeline with gradual ramp-up
        # ROI typically starts lower and increases as AI system matures
        # Calculate number of months from days_to_deployment
        forecast_month_count = max(12, int(np.ceil(request.days_to_deployment / 30)))
        
        forecast_months = []
        for month in range(1, forecast_month_count + 1):
            # Ramp-up curve: starts at 30% of predicted, reaches 100% by month 6, then stabilizes
            if month <= 6:
                ramp_factor = 0.3 + (0.7 * (month / 6))
            else:
                ramp_factor = 1.0 + (0.1 * np.random.normal(0, 0.1))  # Small variation after stabilization
            
            month_roi = predicted_roi * ramp_factor
            month_lower = roi_lower * ramp_factor
            month_upper = roi_upper * ramp_factor
            
            forecast_months.append({
                'month': month,
                'roi': round(month_roi, 2),
                'lower': round(month_lower, 2),
                'upper': round(month_upper, 2)
            })
        
        prob_not_high = float(probabilities[0])
        prob_high = float(probabilities[1])
        # Confidence is the distance from 50% (how decisive the prediction is)
        # 0% = completely uncertain (50/50), 100% = completely certain (0/100 or 100/0)
        confidence = abs(prob_high - 0.5) * 2
        
        # Interpret prediction
        if prediction_binary == 1:
            prediction_label = "High"
            interpretation = f"High ROI Expected (≥145.5%). Probability: {prob_high*100:.1f}% | Confidence: {confidence*100:.1f}%"
        else:
            prediction_label = "Not-High"
            interpretation = f"Not-High ROI Expected (<145.5%). Probability: {prob_not_high*100:.1f}% | Confidence: {confidence*100:.1f}%"
        
        return PredictionResponse(
            prediction=prediction_label,
            probability_high=prob_high,
            probability_not_high=prob_not_high,
            confidence=confidence,
            threshold=145.5,
            interpretation=interpretation,
            predicted_roi=round(predicted_roi, 2),
            roi_lower_bound=round(roi_lower, 2),
            roi_upper_bound=round(roi_upper, 2),
            forecast_months=forecast_months
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
        "classifier_loaded": classifier_model is not None,
        "regression_loaded": regression_model is not None
    }

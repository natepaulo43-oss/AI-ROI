from pydantic import BaseModel, Field
from typing import Optional

class PredictionInput(BaseModel):
    """Input schema for ROI prediction - all fields required by the model"""
    
    # Temporal features
    year: int = Field(..., ge=2020, le=2030, description="Deployment year")
    quarter: str = Field(..., pattern="^(q1|q2|q3|q4)$", description="Deployment quarter")
    
    # Company characteristics
    sector: str = Field(..., description="Industry sector (e.g., manufacturing, finance, retail)")
    company_size: str = Field(..., pattern="^(pme|eti|grande)$", description="Company size category")
    revenue_m_eur: float = Field(..., gt=0, description="Company revenue in millions EUR")
    
    # AI deployment characteristics
    ai_use_case: str = Field(..., description="AI use case type")
    deployment_type: str = Field(..., pattern="^(analytics|nlp|hybrid|automation|vision)$", description="Deployment type")
    
    # Timeline features
    days_diagnostic: int = Field(..., ge=0, description="Days spent in diagnostic phase")
    days_poc: int = Field(..., ge=0, description="Days spent in proof-of-concept")
    days_to_deployment: int = Field(..., ge=1, description="Total days to deployment")
    
    # Investment
    investment_eur: float = Field(..., gt=0, description="Total AI investment in EUR")
    
    # Early deployment signals (can be 0 if not yet available)
    time_saved_hours_month: float = Field(0.0, ge=0, description="Time saved per month (hours)")
    revenue_increase_percent: float = Field(0.0, ge=0, description="Revenue increase percentage")
    
    # Configuration
    human_in_loop: int = Field(..., ge=0, le=1, description="Human oversight (0 or 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "quarter": "q1",
                "sector": "manufacturing",
                "company_size": "grande",
                "revenue_m_eur": 330.7,
                "ai_use_case": "customer service bot",
                "deployment_type": "analytics",
                "days_diagnostic": 35,
                "days_poc": 115,
                "days_to_deployment": 360,
                "investment_eur": 353519,
                "time_saved_hours_month": 0,
                "revenue_increase_percent": 0.0,
                "human_in_loop": 1
            }
        }

class MonthlyForecast(BaseModel):
    """Monthly ROI forecast data point"""
    month: int = Field(..., description="Month number (1-12)")
    roi: float = Field(..., description="Predicted ROI for this month")
    lower: float = Field(..., description="Lower confidence bound")
    upper: float = Field(..., description="Upper confidence bound")

class PredictionOutput(BaseModel):
    """Output schema for ROI prediction - complete response with classification and forecast"""
    prediction: str = Field(..., description="Classification: 'High' or 'Not-High'")
    probability_high: float = Field(..., description="Probability of High ROI (0-1)")
    probability_not_high: float = Field(..., description="Probability of Not-High ROI (0-1)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    threshold: float = Field(..., description="ROI threshold for High classification (145.5%)")
    interpretation: str = Field(..., description="Human-readable interpretation")
    predicted_roi: float = Field(..., description="Predicted ROI percentage")
    roi_lower_bound: float = Field(..., description="Lower confidence interval")
    roi_upper_bound: float = Field(..., description="Upper confidence interval")
    forecast_months: list[MonthlyForecast] = Field(..., description="12-month ROI forecast")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "High",
                "probability_high": 0.72,
                "probability_not_high": 0.28,
                "confidence": 0.72,
                "threshold": 145.5,
                "interpretation": "High ROI Expected (≥145.5%). Confidence: 72.0%",
                "predicted_roi": 178.5,
                "roi_lower_bound": 115.83,
                "roi_upper_bound": 241.17,
                "forecast_months": [
                    {"month": 1, "roi": 53.55, "lower": 34.75, "upper": 72.35},
                    {"month": 12, "roi": 178.5, "lower": 115.83, "upper": 241.17}
                ]
            }
        }

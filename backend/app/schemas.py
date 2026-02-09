from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float = None

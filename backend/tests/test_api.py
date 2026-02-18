"""
Unit tests for the AI ROI Prediction API
Uses FastAPI TestClient to test endpoints without running a server.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "AI ROI Prediction API"
    assert data["version"] == "2.0"
    assert data["status"] == "running"

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_version" in data

def test_prediction_basic():
    """Test the prediction endpoint with valid data"""
    sample_data = {
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
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_roi" in data
    assert "classification" in data
    assert "model_version" in data

def test_prediction_with_signals():
    """Test prediction with early deployment signals"""
    sample_data = {
        "year": 2024,
        "quarter": "q4",
        "sector": "finance",
        "company_size": "eti",
        "revenue_m_eur": 57.2,
        "ai_use_case": "pricing optimization",
        "deployment_type": "analytics",
        "days_diagnostic": 31,
        "days_poc": 64,
        "days_to_deployment": 134,
        "investment_eur": 49189,
        "time_saved_hours_month": 512,
        "revenue_increase_percent": 23.8,
        "human_in_loop": 1
    }
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_roi" in data
    assert isinstance(data["predicted_roi"], (int, float))

def test_invalid_quarter():
    """Test with invalid quarter value"""
    invalid_data = {
        "year": 2024,
        "quarter": "q5",
        "sector": "manufacturing",
        "company_size": "grande",
        "revenue_m_eur": 100.0,
        "ai_use_case": "customer service bot",
        "deployment_type": "analytics",
        "days_diagnostic": 35,
        "days_poc": 115,
        "days_to_deployment": 360,
        "investment_eur": 100000,
        "time_saved_hours_month": 0,
        "revenue_increase_percent": 0.0,
        "human_in_loop": 1
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_missing_required_fields():
    """Test with missing required fields"""
    incomplete_data = {
        "year": 2024,
        "quarter": "q1"
    }
    
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422

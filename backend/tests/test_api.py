"""
Unit tests for the AI ROI Prediction API
Uses FastAPI TestClient to test endpoints without running a server.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from fastapi.testclient import TestClient

def create_mock_models():
    """Create mock models for testing"""
    mock_classifier = Mock()
    mock_classifier.predict.return_value = np.array([1])
    mock_classifier.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    mock_regression = Mock()
    mock_regression.predict.return_value = np.array([180.5])
    
    return {
        'classifier': mock_classifier,
        'regression': mock_regression
    }

@pytest.fixture
def client():
    """Create test client with mocked models"""
    with patch('app.model_loader.load_model', return_value=create_mock_models()):
        # Import app inside the patch context
        from app.main import app
        from app import main as app_main
        
        # Set models to mock
        app_main.models = create_mock_models()
        
        with TestClient(app) as test_client:
            yield test_client

def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "AI ROI Prediction API"
    assert data["version"] == "2.0"
    assert data["status"] == "running"

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_version" in data

def test_prediction_basic(client):
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
    assert "prediction" in data
    assert "probability_high" in data
    assert "forecast_months" in data
    assert isinstance(data["predicted_roi"], (int, float))

def test_prediction_with_signals(client):
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

def test_invalid_quarter(client):
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

def test_missing_required_fields(client):
    """Test with missing required fields"""
    incomplete_data = {
        "year": 2024,
        "quarter": "q1"
    }
    
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422

"""
Test script for the AI ROI Prediction API
Run this after starting the backend server to verify the API works correctly.
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction():
    """Test the prediction endpoint with sample data"""
    print("=" * 60)
    print("TEST 2: ROI Prediction")
    print("=" * 60)
    
    # Sample input data (from the training dataset)
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
    
    print("Input Data:")
    print(json.dumps(sample_data, indent=2))
    print()
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Prediction successful!")
        print(f"Predicted ROI: {result['predicted_roi']:.2f}%")
        print(f"Model Version: {result['model_version']}")
        print(f"Confidence: {result['confidence_note']}")
    else:
        print(f"\n❌ Prediction failed!")
        print(f"Error: {response.text}")
    print()

def test_prediction_with_signals():
    """Test prediction with early deployment signals"""
    print("=" * 60)
    print("TEST 3: Prediction with Early Signals")
    print("=" * 60)
    
    # Sample with early deployment signals
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
        "time_saved_hours_month": 512,  # Early signal
        "revenue_increase_percent": 23.8,  # Early signal
        "human_in_loop": 1
    }
    
    print("Input Data (with early signals):")
    print(json.dumps(sample_data, indent=2))
    print()
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Prediction successful!")
        print(f"Predicted ROI: {result['predicted_roi']:.2f}%")
        print(f"Model Version: {result['model_version']}")
        print(f"Note: This prediction should be more accurate due to early signals")
    else:
        print(f"\n❌ Prediction failed!")
        print(f"Error: {response.text}")
    print()

def test_invalid_input():
    """Test with invalid input to verify validation"""
    print("=" * 60)
    print("TEST 4: Invalid Input Validation")
    print("=" * 60)
    
    invalid_data = {
        "year": 2024,
        "quarter": "q5",  # Invalid quarter
        "sector": "manufacturing",
        "company_size": "grande",
        "revenue_m_eur": -100,  # Invalid negative revenue
    }
    
    print("Invalid Input Data:")
    print(json.dumps(invalid_data, indent=2))
    print()
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 422:
        print(f"\n✅ Validation working correctly!")
        print(f"Validation errors detected:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n⚠️ Unexpected response")
        print(f"Response: {response.text}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AI ROI PREDICTION API - TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")
    print()
    
    try:
        # Run all tests
        test_health_check()
        test_prediction()
        test_prediction_with_signals()
        test_invalid_input()
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the backend server is running:")
        print("  cd backend")
        print("  uvicorn app.main:app --reload")
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

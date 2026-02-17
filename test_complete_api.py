"""
Test script to verify the complete API returns all required data for the frontend.
Tests both model loading and prediction with full response validation.
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    data = response.json()
    assert data['status'] == 'healthy', "API not healthy"
    assert data['classifier_loaded'], "Classifier not loaded"
    assert data['regression_loaded'], "Regression model not loaded"
    print("✓ Health check passed\n")

def test_prediction():
    """Test prediction endpoint with sample data"""
    print("=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)
    
    # Sample request matching the frontend format
    request_data = {
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
        "time_saved_hours_month": 150.0,
        "revenue_increase_percent": 5.0,
        "human_in_loop": 1
    }
    
    print(f"Request Data:")
    print(json.dumps(request_data, indent=2))
    print()
    
    response = requests.post(f"{API_URL}/predict", json=request_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return False
    
    data = response.json()
    print(f"\nResponse Data:")
    print(json.dumps(data, indent=2))
    
    # Validate all required fields are present
    required_fields = [
        'prediction', 'probability_high', 'probability_not_high',
        'confidence', 'threshold', 'interpretation',
        'predicted_roi', 'roi_lower_bound', 'roi_upper_bound',
        'forecast_months'
    ]
    
    print("\n" + "=" * 60)
    print("Validating Response Fields")
    print("=" * 60)
    
    for field in required_fields:
        if field in data:
            print(f"✓ {field}: {type(data[field]).__name__}")
        else:
            print(f"✗ MISSING: {field}")
            return False
    
    # Validate forecast_months structure
    print("\n" + "=" * 60)
    print("Validating Forecast Data")
    print("=" * 60)
    
    forecast = data['forecast_months']
    print(f"Forecast length: {len(forecast)} months")
    
    if len(forecast) != 12:
        print(f"✗ Expected 12 months, got {len(forecast)}")
        return False
    
    # Check first and last month
    first_month = forecast[0]
    last_month = forecast[-1]
    
    print(f"\nMonth 1: ROI={first_month['roi']}%, Range=[{first_month['lower']}%, {first_month['upper']}%]")
    print(f"Month 12: ROI={last_month['roi']}%, Range=[{last_month['lower']}%, {last_month['upper']}%]")
    
    # Validate each month has required fields
    for month_data in forecast:
        required_month_fields = ['month', 'roi', 'lower', 'upper']
        for field in required_month_fields:
            if field not in month_data:
                print(f"✗ MISSING in forecast: {field}")
                return False
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Classification: {data['prediction']}")
    print(f"Confidence: {data['confidence']*100:.1f}%")
    print(f"Predicted ROI: {data['predicted_roi']}%")
    print(f"Confidence Interval: [{data['roi_lower_bound']}%, {data['roi_upper_bound']}%]")
    print(f"Interpretation: {data['interpretation']}")
    print(f"\n✓ All validation checks passed!")
    
    return True

if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("AI ROI API - Complete System Test")
        print("=" * 60)
        print()
        
        # Test health
        test_health()
        
        # Test prediction
        success = test_prediction()
        
        if success:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED - API IS READY FOR FRONTEND")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ TESTS FAILED - CHECK ERRORS ABOVE")
            print("=" * 60)
            
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to API at http://localhost:8000")
        print("Please start the backend server first:")
        print("  cd backend")
        print("  uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

import requests
import json

# Test the updated API with the best binary classifier model

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("TESTING UPDATED API WITH BEST BINARY CLASSIFIER")
print("=" * 80)

# Test 1: Health check
print("\n1. Testing health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")
    print("   Make sure the API is running: uvicorn backend.roi_api:app --reload")

# Test 2: Prediction with sample data (High ROI scenario)
print("\n2. Testing prediction endpoint (High ROI scenario)...")
high_roi_data = {
    "investment_eur": 50000,
    "revenue_m_eur": 10.0,
    "human_in_loop": 1,
    "days_to_deployment": 120,
    "days_diagnostic": 20,
    "days_poc": 40,
    "time_saved_hours_month": 500,
    "revenue_increase_percent": 15.0,
    "year": 2024,
    "quarter": "q1",
    "sector": "technology",
    "company_size": "pme",
    "ai_use_case": "predictive analytics",
    "deployment_type": "cloud"
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=high_roi_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n   PREDICTION RESULTS:")
        print(f"   - Prediction: {result['prediction']}")
        print(f"   - Probability High ROI: {result['probability_high']*100:.2f}%")
        print(f"   - Probability Not-High: {result['probability_not_high']*100:.2f}%")
        print(f"   - Confidence: {result['confidence']*100:.2f}%")
        print(f"   - Threshold: {result['threshold']}%")
        print(f"   - Interpretation: {result['interpretation']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Prediction with sample data (Low ROI scenario)
print("\n3. Testing prediction endpoint (Low ROI scenario)...")
low_roi_data = {
    "investment_eur": 500000,
    "revenue_m_eur": 2.0,
    "human_in_loop": 0,
    "days_to_deployment": 300,
    "days_diagnostic": 50,
    "days_poc": 100,
    "time_saved_hours_month": 0,
    "revenue_increase_percent": 0,
    "year": 2024,
    "quarter": "q1",
    "sector": "construction",
    "company_size": "grande",
    "ai_use_case": "process automation",
    "deployment_type": "on-premise"
}

try:
    response = requests.post(f"{BASE_URL}/predict", json=low_roi_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n   PREDICTION RESULTS:")
        print(f"   - Prediction: {result['prediction']}")
        print(f"   - Probability High ROI: {result['probability_high']*100:.2f}%")
        print(f"   - Probability Not-High: {result['probability_not_high']*100:.2f}%")
        print(f"   - Confidence: {result['confidence']*100:.2f}%")
        print(f"   - Threshold: {result['threshold']}%")
        print(f"   - Interpretation: {result['interpretation']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("API TESTING COMPLETE")
print("=" * 80)
print("\nNOTE: If you see connection errors, start the API with:")
print("  uvicorn backend.roi_api:app --reload")
print("\nThe API now uses the best binary classifier (68.8% accuracy, p < 0.001)")

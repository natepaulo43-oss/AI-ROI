import requests
import json

# Test the API with a sample request
test_data = {
    "investment_eur": 100000,
    "revenue_m_eur": 10,
    "human_in_loop": 1,
    "days_to_deployment": 90,
    "days_diagnostic": 15,
    "days_poc": 30,
    "time_saved_hours_month": 100,
    "revenue_increase_percent": 5,
    "year": 2024,
    "quarter": "q2",
    "sector": "finance",
    "company_size": "grande",
    "ai_use_case": "automation",
    "deployment_type": "hybrid"
}

try:
    response = requests.post('http://localhost:8000/predict', json=test_data)
    print("Status Code:", response.status_code)
    print("\nResponse JSON:")
    result = response.json()
    print(json.dumps(result, indent=2))
    print("\n=== Key Values ===")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Probability High: {result.get('probability_high')}")
    print(f"Probability Not-High: {result.get('probability_not_high')}")
except Exception as e:
    print(f"Error: {e}")

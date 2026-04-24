from fastapi.testclient import TestClient
try:
    from api.main import app
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from api.main import app

client = TestClient(app)

valid_payload = {
    "AccountWeeks": 100,
    "ContractRenewal": 1,
    "DataPlan": 0,
    "DataUsage": 0,
    "CustServCalls": 1,
    "DayMins": 200,
    "DayCalls": 100,
    "MonthlyCharge": 50,
    "OverageFee": 10,
    "RoamMins": 10
}

print("\n--- Testing /predict (Valid) ---")
r_predict = client.post("/predict", json=valid_payload)
print(f"Status: {r_predict.status_code}")
print(f"JSON: {r_predict.json()}")

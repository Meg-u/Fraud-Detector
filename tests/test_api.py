from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_credit_works():
    payload = {"features": {"Time":0,"Amount":10.0, **{f"V{i}":0.0 for i in range(1,29)}}}
    res = client.post("/predict/credit", json=payload)
    assert res.status_code == 200
    assert "probability" in res.json()

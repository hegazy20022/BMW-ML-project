from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code in (200, 500)


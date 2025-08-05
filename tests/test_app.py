from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1. Test /predict with valid input
def test_prediction():
    response = client.post("/predict", json={
        "title": "NASA announces new moon mission",
        "description": "The space agency will send astronauts back to the moon by 2025."
    })
    assert response.status_code == 200
    assert "predicted_class" in response.json()

# 2. Test /predict with missing fields
def test_prediction_missing_field():
    response = client.post("/predict", json={
        "title": "Missing description"
    })
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)

# 3. Test /predict with empty input
def test_prediction_empty_input():
    response = client.post("/predict", json={
        "title": "",
        "description": ""
    })
    assert response.status_code == 200
    assert "predicted_class" in response.json()

# 4. Test root route
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

# 5. Test /model-info
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    json_data = response.json()
    assert "model_name" in json_data
    assert "labels" in json_data

# 6. Test invalid route
def test_invalid_route():
    response = client.get("/invalid-route")
    assert response.status_code == 404

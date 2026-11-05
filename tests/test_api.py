"""Feature: FastAPI service (no Ollama needed)."""
import pytest
from fastapi.testclient import TestClient

from api import app

GOOD = {"no_of_dependents": 2, "education": "Graduate", "self_employed": "No", "income_annum": 5_000_000,
        "loan_amount": 12_000_000, "loan_term": 10, "cibil_score": 780, "residential_assets_value": 4_000_000,
        "commercial_assets_value": 2_000_000, "luxury_assets_value": 8_000_000, "bank_asset_value": 3_000_000}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # runs the lifespan (loads the model once)
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True


def test_predict_returns_prediction_and_certainty(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("Approved", "Rejected") and 0 <= body["probability"] <= 1
    assert len(body["factors"]) == 3 and body["certainty"]["level"] in ("high", "medium", "low")


def test_invalid_input_is_422(client):
    assert client.post("/predict", json={**GOOD, "cibil_score": 1000}).status_code == 422
    assert client.post("/predict", json={**GOOD, "education": "PhD"}).status_code == 422
    missing = dict(GOOD); missing.pop("loan_term")
    assert client.post("/predict", json=missing).status_code == 422


def test_what_it_takes(client):
    r = client.post("/what-it-takes", json={**GOOD, "cibil_score": 400}).json()
    assert not r["already"] and r["plans"]
    assert client.post("/what-it-takes?target=0.3", json=GOOD).status_code == 422


def test_explain_works_without_ollama(client, monkeypatch):
    monkeypatch.setattr("loanlens.llm.ollama_reachable", lambda timeout=0.5: False)
    r = client.post("/explain", json={**GOOD, "cibil_score": 400}).json()
    assert r["prediction"]["label"] == "Rejected" and r["sources"]
    assert r["explanation"]["status"] == "ollama_unavailable"

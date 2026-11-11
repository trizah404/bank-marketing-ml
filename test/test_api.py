from fastapi.testclient import TestClient

from app.api import app, EXAMPLE

client = TestClient(app)          # a fake browser; no server needs to be running


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_valid_customer_gets_prediction():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["call"] == (body["probability"] >= body["threshold"])


def test_rejects_leaky_duration():
    bad = {**EXAMPLE, "duration": 300}        # copy EXAMPLE, add duration
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_unknown_job():
    bad = {**EXAMPLE, "job": "astronaut"}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_missing_field():
    bad = {k: v for k, v in EXAMPLE.items() if k != "age"}   # remove age
    assert client.post("/predict", json=bad).status_code == 422
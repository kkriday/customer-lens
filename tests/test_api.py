"""API endpoint tests using FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

import customerlens.api as api_module
from customerlens.api import app


@pytest.fixture()
def client(engine):
    # Inject the fixture engine into the app module (isolated per test).
    api_module.engine = engine
    yield TestClient(app)
    api_module.engine = api_module.Engine()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["ready"] is True


def test_profile(client):
    r = client.get("/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["tickets"]["rows"] == 5
    assert body["tickets"]["duplicate_rows"] == 1


def test_ask(client):
    r = client.post("/ask", json={"question": "Atlas CRM outage ERR-9001", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "extractive"
    assert body["citations"][0]["ticket_id"] == "T-3"


def test_ask_before_ingest():
    fresh = TestClient(app)  # module engine was reset by fixture teardown
    r = fresh.post("/ask", json={"question": "hello"})
    assert r.status_code == 409


def test_ingest_endpoint(tmp_path, tickets_df, customers_df, client):
    tp = tmp_path / "tickets.csv"
    cp = tmp_path / "customers.csv"
    tickets_df.to_csv(tp, index=False)
    customers_df.to_csv(cp, index=False)
    r = client.post("/ingest", json={
        "tickets_path": str(tp), "customers_path": str(cp),
        "artifact_dir": str(tmp_path / "artifacts")})
    assert r.status_code == 200
    body = r.json()
    assert body["tickets"] == 4
    assert body["status"] == "ingested"


def test_ingest_missing_file(client, tmp_path):
    r = client.post("/ingest", json={
        "tickets_path": str(tmp_path / "nope.csv"),
        "customers_path": str(tmp_path / "nope.csv"),
        "artifact_dir": str(tmp_path / "artifacts")})
    assert r.status_code == 400

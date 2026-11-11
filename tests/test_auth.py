import pytest
from fastapi.testclient import TestClient

from src.api import observations
from src.api.main import app

OBS = {"crop": "groundnut", "lat": 11.9, "lon": 79.5, "observed_date": "2026-01-01",
       "problem": "none", "severity": "none", "reporter_role": "farmer"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(observations, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setenv("AGRIRISK_API_KEY", "secret-for-tests")
    return TestClient(app)


def test_get_requires_key(client):
    assert client.get("/observations").status_code == 401


def test_post_requires_key(client):
    assert client.post("/observations", json=OBS).status_code == 401


def test_wrong_key_rejected(client):
    assert client.get("/observations", headers={"X-API-Key": "nope"}).status_code == 401


def test_unset_key_disables_endpoint(client, monkeypatch):
    monkeypatch.delenv("AGRIRISK_API_KEY")
    assert client.get("/observations", headers={"X-API-Key": "anything"}).status_code == 503


def test_correct_key_works_and_other_routes_unaffected(client):
    h = {"X-API-Key": "secret-for-tests"}
    assert client.post("/observations", json=OBS, headers=h).status_code == 201
    assert client.get("/observations", headers=h).status_code == 200
    assert client.get("/models").status_code == 200

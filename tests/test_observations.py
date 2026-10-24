import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.api.observations as obs_api
from src.api.main import app
from src.observations.schema import ObservationIn

BASE = {"crop": "groundnut", "lat": 11.99, "lon": 79.49,
        "observed_date": "2026-10-01", "problem": "waterlogging",
        "severity": "moderate", "reporter_role": "farmer"}


def test_valid_minimal():
    assert ObservationIn(**BASE).diagnosis_basis == "reporter_observation"


def test_future_date_rejected():
    with pytest.raises(ValidationError):
        ObservationIn(**{**BASE, "observed_date": "2999-01-01"})


def test_none_problem_requires_none_severity():
    with pytest.raises(ValidationError):
        ObservationIn(**{**BASE, "problem": "none"})
    ObservationIn(**{**BASE, "problem": "none", "severity": "none"})


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        ObservationIn(**{**BASE, "phone": "123"})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(obs_api, "DB_PATH", tmp_path / "o.db")
    return TestClient(app)


def test_post_and_list(client):
    r = client.post("/observations", json=BASE)
    assert r.status_code == 201 and r.json()["id"]
    assert len(client.get("/observations").json()) == 1


def test_filter_by_crop(client):
    client.post("/observations", json=BASE)
    client.post("/observations", json={**BASE, "crop": "sesame"})
    got = client.get("/observations", params={"crop": "sesame"}).json()
    assert len(got) == 1 and got[0]["crop"] == "sesame"


def test_invalid_crop_422(client):
    assert client.post("/observations", json={**BASE, "crop": "wheat"}).status_code == 422

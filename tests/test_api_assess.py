from fastapi.testclient import TestClient

import src.api.assess as assess_module
from src.api.main import app
from src.data_collection.soilgrids_client import SoilGridsUnavailableError

client = TestClient(app)


def post(**kw):
    body = {"crop": "groundnut", "lat": 11.99, "lon": 79.49, "include_soil": False}
    body.update(kw)
    return client.post("/assess", json=body)


def level(r, cat):
    return next(a for a in r.json()["assessments"] if a["category"] == cat)


def test_default_all_incomplete():
    r = post()
    assert r.status_code == 200
    assert all(a["status"] == "incomplete" for a in r.json()["assessments"])
    assert r.json()["soil"]["status"] == "not_requested"


def test_heat_elevated_with_observation():
    r = post(observations={"max_temperature_c": 42})
    assert level(r, "heat_stress")["level"] == "elevated"


def test_unknown_crop_422():
    assert post(crop="wheat").status_code == 422


def test_negative_water_days_422():
    assert post(observations={"standing_water_days": -1}).status_code == 422


def test_bad_coordinate_422():
    assert post(lat=95).status_code == 422


def test_soil_unavailable_still_200(monkeypatch):
    def boom(*a, **k):
        raise SoilGridsUnavailableError("network down")
    monkeypatch.setattr(assess_module, "fetch_soil_properties", boom)
    r = post(include_soil=True)
    assert r.status_code == 200
    assert r.json()["soil"]["status"] == "unavailable"

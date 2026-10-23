from fastapi.testclient import TestClient

import src.api.assess as assess_module
from src.api.main import app
from src.data_collection.soilgrids_client import SoilGridsUnavailableError

client = TestClient(app)


def post(**kw):
    body = {"crop": "groundnut", "lat": 11.99, "lon": 79.49, "include_soil": False,
            "include_weather": False}
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


def _fake_soil(clay_fails=False):
    from src.data_collection.soilgrids_client import Measurement, SoilResult
    ms = [
        Measurement("sand", "0-5cm", "mean", 399.0, 39.9, "%", "ok", read_via="local"),
        Measurement("silt", "0-5cm", "mean", 275.0, 27.5, "%", "ok", read_via="local"),
        Measurement("phh2o", "0-5cm", "mean", 70.0, 7.0, "pH (H2O)", "ok", read_via="local"),
    ]
    if clay_fails:
        ms.append(Measurement("clay", "0-5cm", "mean", None, None, "%", "error",
                              "Access error: Cannot read 5928942 bytes"))
    else:
        ms.append(Measurement("clay", "0-5cm", "mean", 327.0, 32.7, "%", "ok",
                              read_via="local"))
    return SoilResult(11.99, 79.49, "test", "2026-10-23T00:00:00+00:00", 250, ms)


def test_soil_partial_names_failed_layer(monkeypatch):
    monkeypatch.setattr(assess_module, "fetch_soil_properties",
                        lambda *a, **k: _fake_soil(clay_fails=True))
    r = post(include_soil=True)
    soil = r.json()["soil"]
    assert r.status_code == 200
    assert soil["status"] == "partial" and soil["layers_ok"] == 3
    assert "clay_0-5cm" in soil["message"]
    assert "Cannot read 5928942 bytes" in soil["message"]
    assert soil["read_via"] == ["local"]


def test_soil_ok_has_empty_message_and_read_via(monkeypatch):
    monkeypatch.setattr(assess_module, "fetch_soil_properties",
                        lambda *a, **k: _fake_soil())
    soil = post(include_soil=True).json()["soil"]
    assert soil["status"] == "ok" and soil["layers_ok"] == 4
    assert soil["message"] == "" and soil["read_via"] == ["local"]

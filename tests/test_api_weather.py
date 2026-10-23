from fastapi.testclient import TestClient

import src.api.assess as assess_module
from src.api.main import app
from src.data_collection.weather_client import (
    DailyWeather, WeatherResult, WeatherUnavailableError)

client = TestClient(app)


def fake_weather(maxes, status="ok", message=""):
    days = [DailyWeather(date=f"2026-10-{16 + i}", max_temperature_c=m,
                         precipitation_mm=1.0, et0_mm=3.0)
            for i, m in enumerate(maxes)]
    return WeatherResult(
        requested_lat=11.99, requested_lon=79.49,
        grid_lat=11.98594, grid_lon=79.49731, elevation_m=42.0,
        days=days, status=status, message=message,
        source="open-meteo (modelled gridded estimate, not a station measurement)",
        retrieved_at_utc="2026-10-23T16:20:09+00:00")


def patch_weather(monkeypatch, result=None, error=None):
    def fake(lat, lon, days=7, today=None):
        if error:
            raise error
        return result
    monkeypatch.setattr(assess_module, "fetch_recent_weather", fake)


def post(**kw):
    body = {"crop": "groundnut", "lat": 11.99, "lon": 79.49,
            "include_soil": False}
    body.update(kw)
    return client.post("/assess", json=body)


def heat(r):
    return next(a for a in r.json()["assessments"]
                if a["category"] == "heat_stress")


def test_hottest_day_used_and_labelled_retrieved(monkeypatch):
    patch_weather(monkeypatch, fake_weather([31.0, 36.5, 33.0]))
    r = post()
    assert r.status_code == 200
    h = heat(r)
    assert h["status"] == "assessed"
    assert h["inputs_used"]["max_temperature_c"] == {
        "value": 36.5, "provenance": "retrieved"}
    assert "2026-10-17" in h["reasons"][0]
    assert any("design choice" in c for c in h["caveats"])
    w = r.json()["weather"]
    assert w["status"] == "ok"
    assert w["hottest_day"] == "2026-10-17"
    assert w["max_temperature_c"] == 36.5
    assert w["grid_lat"] == 11.98594 and w["elevation_m"] == 42.0


def test_manual_observation_wins_over_weather(monkeypatch):
    patch_weather(monkeypatch, fake_weather([31.0, 32.0]))
    r = post(observations={"max_temperature_c": 44})
    h = heat(r)
    assert h["level"] == "elevated"
    assert h["inputs_used"]["max_temperature_c"]["provenance"] == "manual"


def test_weather_unavailable_still_200(monkeypatch):
    patch_weather(monkeypatch, error=WeatherUnavailableError("network down"))
    r = post()
    assert r.status_code == 200
    assert r.json()["weather"]["status"] == "unavailable"
    assert heat(r)["status"] == "incomplete"


def test_weather_not_requested(monkeypatch):
    patch_weather(monkeypatch, error=AssertionError("must not be called"))
    r = post(include_weather=False)
    assert r.status_code == 200
    assert r.json()["weather"]["status"] == "not_requested"
    assert heat(r)["status"] == "incomplete"


def test_missing_temperatures_stay_incomplete(monkeypatch):
    patch_weather(monkeypatch, fake_weather([None, None], status="nodata",
                                            message="no temperature values"))
    r = post()
    w = r.json()["weather"]
    assert w["status"] == "nodata"
    assert w["days_ok"] == 0 and w["max_temperature_c"] is None
    assert heat(r)["status"] == "incomplete"


def test_partial_weather_uses_days_that_exist(monkeypatch):
    patch_weather(monkeypatch, fake_weather([None, 34.0, 30.0],
                                            status="partial",
                                            message="1 day missing temperature"))
    r = post()
    w = r.json()["weather"]
    assert w["status"] == "partial" and w["days_ok"] == 2
    assert heat(r)["inputs_used"]["max_temperature_c"]["value"] == 34.0

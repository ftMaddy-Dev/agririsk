import json
import urllib.error
from datetime import date

import pytest

from src.data_collection import weather_client as wc

TODAY = date(2026, 10, 23)
DATES = ["2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19",
         "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]


def payload(tmax=None, rain=None, et0=None, dates=DATES):
    n = len(dates)
    return {
        "latitude": 11.98594, "longitude": 79.49731, "elevation": 42.0,
        "daily": {
            "time": dates,
            "temperature_2m_max": tmax if tmax is not None else [31.0] * n,
            "precipitation_sum": rain if rain is not None else [1.0] * n,
            "et0_fao_evapotranspiration": et0 if et0 is not None else [3.0] * n,
        },
    }


def test_excludes_today_and_keeps_completed_days(monkeypatch):
    monkeypatch.setattr(wc, "_http_get_json", lambda url: payload())
    r = wc.fetch_recent_weather(11.99, 79.49, days=7, today=TODAY)
    assert len(r.days) == 7
    assert r.days[-1].date == "2026-10-22"
    assert r.status == "ok" and r.message == ""
    assert r.grid_lat == 11.98594 and r.elevation_m == 42.0


def test_missing_values_stay_none_not_zero(monkeypatch):
    p = payload(et0=[3.0, None, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                tmax=[31.0, 31.0, None, 31.0, 31.0, 31.0, 31.0, 31.0])
    monkeypatch.setattr(wc, "_http_get_json", lambda url: p)
    r = wc.fetch_recent_weather(11.99, 79.49, today=TODAY)
    assert r.days[1].et0_mm is None
    assert r.days[2].max_temperature_c is None
    assert r.status == "partial"
    assert "et0_mm: 1 of 7" in r.message
    assert "max_temperature_c: 1 of 7" in r.message


def test_all_null_is_nodata(monkeypatch):
    n = len(DATES)
    p = payload(tmax=[None] * n, rain=[None] * n, et0=[None] * n)
    monkeypatch.setattr(wc, "_http_get_json", lambda url: p)
    r = wc.fetch_recent_weather(11.99, 79.49, today=TODAY)
    assert r.status == "nodata"


def test_only_today_returned_is_nodata(monkeypatch):
    monkeypatch.setattr(wc, "_http_get_json",
                        lambda url: payload(dates=["2026-10-23"]))
    r = wc.fetch_recent_weather(11.99, 79.49, days=1, today=TODAY)
    assert r.days == [] and r.status == "nodata"


@pytest.mark.parametrize("lat,lon,days", [
    (95, 79.49, 7), (11.99, 200, 7), (11.99, 79.49, 0), (11.99, 79.49, 93),
    (float("nan"), 79.49, 7)])
def test_invalid_inputs_raise_value_error(lat, lon, days):
    with pytest.raises(ValueError):
        wc.fetch_recent_weather(lat, lon, days=days, today=TODAY)


class FakeResp:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("boom")
        return FakeResp(payload())

    monkeypatch.setattr(wc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wc.time, "sleep", lambda s: None)
    r = wc.fetch_recent_weather(11.99, 79.49, today=TODAY)
    assert calls["n"] == 3 and r.status == "ok"


def test_unavailable_after_all_attempts(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("down")

    monkeypatch.setattr(wc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wc.time, "sleep", lambda s: None)
    with pytest.raises(wc.WeatherUnavailableError):
        wc.fetch_recent_weather(11.99, 79.49, today=TODAY)
    assert calls["n"] == 3


def test_client_error_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 400, "bad", {}, None)

    monkeypatch.setattr(wc.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(wc.WeatherError):
        wc.fetch_recent_weather(11.99, 79.49, today=TODAY)
    assert calls["n"] == 1


def test_unexpected_payload_is_unavailable(monkeypatch):
    monkeypatch.setattr(wc, "_http_get_json", lambda url: {"oops": 1})
    with pytest.raises(wc.WeatherUnavailableError):
        wc.fetch_recent_weather(11.99, 79.49, today=TODAY)

import json
from datetime import datetime, timedelta, timezone

import pytest
from rasterio.errors import RasterioIOError

from src.data_collection import soilgrids_client as sg
from src.data_collection.soilgrids_client import Measurement


def _ok():
    return Measurement("phh2o", "0-5cm", "mean", 70.0, 7.0, "pH (H2O)", "ok")


def _nodata():
    return Measurement("phh2o", "0-5cm", "mean", -32768.0, None, "pH (H2O)", "nodata")


def _read():
    return sg._read_layer("phh2o", "0-5cm", "mean", 11.99, 79.49)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sg.time, "sleep", lambda s: None)


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}
    def fake(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RasterioIOError("Cannot read 123 bytes")
        return _ok()
    monkeypatch.setattr(sg, "_sample_once", fake)
    m = _read()
    assert m.status == "ok" and m.value == 7.0 and calls["n"] == 3


def test_failure_is_not_cached(monkeypatch):
    def fake(*a, **k):
        raise RasterioIOError("boom")
    monkeypatch.setattr(sg, "_sample_once", fake)
    m = _read()
    assert m.status == "error" and m.value is None
    assert list(sg.CACHE_DIR.glob("*.json")) == []


def test_ok_is_cached_and_reused(monkeypatch):
    calls = {"n": 0}
    def fake(*a, **k):
        calls["n"] += 1
        return _ok()
    monkeypatch.setattr(sg, "_sample_once", fake)
    _read()
    second = _read()
    assert calls["n"] == 1 and second.value == 7.0


def test_nodata_cached_and_stays_none(monkeypatch):
    monkeypatch.setattr(sg, "_sample_once", lambda *a, **k: _nodata())
    _read()
    again = _read()
    assert again.status == "nodata" and again.value is None


def test_expired_entry_is_refetched(monkeypatch):
    calls = {"n": 0}
    def fake(*a, **k):
        calls["n"] += 1
        return _ok()
    monkeypatch.setattr(sg, "_sample_once", fake)
    _read()
    f = next(sg.CACHE_DIR.glob("*.json"))
    data = json.loads(f.read_text())
    data["cached_at_utc"] = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    f.write_text(json.dumps(data))
    _read()
    assert calls["n"] == 2


def test_use_cache_false_skips_cache(monkeypatch):
    monkeypatch.setattr(sg, "_sample_once", lambda *a, **k: _ok())
    sg._read_layer("phh2o", "0-5cm", "mean", 11.99, 79.49, use_cache=False)
    assert list(sg.CACHE_DIR.glob("*.json")) == []

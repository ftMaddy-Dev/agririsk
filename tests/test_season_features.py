import datetime as dt
import math
import pytest
from src.data_collection.weather_client import InvalidWeatherRequestError, WeatherUnavailableError
from src.data_collection.weather_history import _parse_archive, _validate
from src.preprocessing.season_features import compute_season_features, season_window

START, END = season_window(2019, "06-15", 10)


def days(over=None, n=10):
    base = dt.date(2019, 6, 15)
    rows = []
    for i in range(n):
        rows.append({"date": (base + dt.timedelta(days=i)).isoformat(),
                     "tmax_c": 30.0, "tmin_c": 22.0, "precip_mm": 2.0, "et0_mm": 4.0})
    for i, kv in (over or {}).items():
        rows[i].update(kv)
    return rows


def test_window_dates():
    assert START == dt.date(2019, 6, 15) and END == dt.date(2019, 6, 24)
    s, e = season_window(2025, "06-15", 105)
    assert e == dt.date(2025, 9, 27)


def test_full_features():
    f = compute_season_features(days(), START, END, hot_threshold_c=40)
    assert f["hot_days"] == 0 and f["rain_sum_mm"] == 20.0
    assert f["et0_sum_mm"] == 40.0 and f["rain_minus_et0_mm"] == -20.0
    assert f["tmax_mean_c"] == 30.0


def test_hot_days_counted():
    d = days({1: {"tmax_c": 41.0}, 2: {"tmax_c": 42.5}})
    f = compute_season_features(d, START, END, hot_threshold_c=40)
    assert f["hot_days"] == 2 and f["tmax_max_c"] == 42.5


def test_missing_never_zero():
    d = days({3: {"precip_mm": None}})
    f = compute_season_features(d, START, END, 40)
    assert math.isnan(f["rain_sum_mm"]) and math.isnan(f["rain_minus_et0_mm"])
    assert f["tmax_mean_c"] == 30.0          # 100% tmax coverage, still computed
    allnone = days({i: {"precip_mm": None} for i in range(10)})
    assert math.isnan(compute_season_features(allnone, START, END, 40)["rain_sum_mm"])


def test_low_coverage_gives_nan():
    d = days({i: {"tmax_c": None} for i in range(3)})   # 70% coverage < 90%
    f = compute_season_features(d, START, END, 40)
    assert math.isnan(f["tmax_mean_c"]) and f["tmax_coverage"] == 0.7


def test_threshold_unavailable_gives_nan_hot_days():
    assert math.isnan(compute_season_features(days(), START, END, None)["hot_days"])


def test_archive_validation():
    today = dt.date(2026, 11, 6)
    with pytest.raises(InvalidWeatherRequestError):
        _validate(11.9, 79.4, dt.date(2026, 11, 1), dt.date(2026, 11, 6), today)   # not completed
    with pytest.raises(InvalidWeatherRequestError):
        _validate(95, 79.4, dt.date(2025, 1, 1), dt.date(2025, 1, 5), today)
    with pytest.raises(InvalidWeatherRequestError):
        _validate(11.9, 79.4, dt.date(2025, 1, 5), dt.date(2025, 1, 1), today)
    assert _validate(11.9, 79.4, dt.date(2025, 1, 1), dt.date(2025, 1, 5), today) == (11.9, 79.4)


def test_archive_parse_keeps_none():
    payload = {"daily": {"time": ["2025-01-01", "2025-01-02"],
                         "temperature_2m_max": [30.0, None], "temperature_2m_min": [20.0, 21.0],
                         "precipitation_sum": [0.0, None], "et0_fao_evapotranspiration": [3.0, 3.5]}}
    r = _parse_archive(payload, 11.9, 79.4)
    assert r["days"][1]["tmax_c"] is None and r["days"][1]["precip_mm"] is None
    assert r["days"][0]["precip_mm"] == 0.0     # a real zero stays zero
    with pytest.raises(WeatherUnavailableError):
        _parse_archive({"daily": {"time": []}}, 1, 1)

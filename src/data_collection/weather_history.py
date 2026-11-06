"""Historical daily weather for a past date range (Open-Meteo archive API).

Missing values stay None, never 0. Completed days only. Values are modelled
grid estimates, not station readings.
"""
from __future__ import annotations

import datetime as dt
import math
from urllib.parse import urlencode

from src.data_collection.weather_client import (
    InvalidWeatherRequestError, WeatherUnavailableError, _http_get_json)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["temperature_2m_max", "temperature_2m_min",
              "precipitation_sum", "et0_fao_evapotranspiration"]
ARCHIVE_LAG_DAYS = 2      # archive data lags real time; stay clear of it
MAX_SPAN_DAYS = 366       # assumption: keep each request to about a year


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _validate(lat, lon, start, end, today=None):
    today = today or dt.date.today()
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise InvalidWeatherRequestError("lat/lon must be numbers") from exc
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise InvalidWeatherRequestError("lat/lon out of range")
    if start > end:
        raise InvalidWeatherRequestError("start date is after end date")
    if (end - start).days + 1 > MAX_SPAN_DAYS:
        raise InvalidWeatherRequestError(f"range longer than {MAX_SPAN_DAYS} days")
    if end > today - dt.timedelta(days=ARCHIVE_LAG_DAYS):
        raise InvalidWeatherRequestError("end date must be a completed day, at least "
                                         f"{ARCHIVE_LAG_DAYS} days before today")
    return lat, lon


def _parse_archive(payload: dict, lat: float, lon: float) -> dict:
    daily = (payload or {}).get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise WeatherUnavailableError("archive response has no daily block")
    for var in DAILY_VARS:
        if var not in daily:
            raise WeatherUnavailableError(f"archive response lacks {var}")
        if len(daily[var]) != len(daily["time"]):
            raise WeatherUnavailableError(f"{var} length differs from time")
    days = [{"date": t,
             "tmax_c": _num(daily["temperature_2m_max"][i]),
             "tmin_c": _num(daily["temperature_2m_min"][i]),
             "precip_mm": _num(daily["precipitation_sum"][i]),
             "et0_mm": _num(daily["et0_fao_evapotranspiration"][i])}
            for i, t in enumerate(daily["time"])]
    return {"latitude": lat, "longitude": lon,
            "source": "Open-Meteo archive (modelled reanalysis grid estimate, not a station reading)",
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "days": days}


def fetch_weather_range(lat: float, lon: float, start: dt.date, end: dt.date) -> dict:
    lat, lon = _validate(lat, lon, start, end)
    params = {"latitude": lat, "longitude": lon,
              "start_date": start.isoformat(), "end_date": end.isoformat(),
              "daily": ",".join(DAILY_VARS), "timezone": "Asia/Kolkata"}
    payload = _http_get_json(f"{ARCHIVE_URL}?{urlencode(params)}")
    return _parse_archive(payload, lat, lon)

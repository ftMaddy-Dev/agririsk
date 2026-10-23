"""Open-Meteo daily weather for a point.

Values are a modelled gridded estimate, not a station measurement.
Missing values stay None, never 0. Only completed days are returned;
today's partial day is excluded. There is no cache, so errors are never cached.
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
DAILY_VARS = ["temperature_2m_max", "precipitation_sum",
              "et0_fao_evapotranspiration"]
SOURCE = "open-meteo (modelled gridded estimate, not a station measurement)"
MAX_DAYS = 92
_MAX_ATTEMPTS = 3
_RETRY_PAUSE_S = 1
_TIMEOUT_S = 15


class WeatherError(Exception):
    """Base class for weather client errors."""


class InvalidWeatherRequestError(WeatherError, ValueError):
    """Bad coordinates or day count."""


class WeatherUnavailableError(WeatherError):
    """Provider unreachable or returned something unusable."""


@dataclass
class DailyWeather:
    date: str
    max_temperature_c: float | None = None
    precipitation_mm: float | None = None
    et0_mm: float | None = None


@dataclass
class WeatherResult:
    requested_lat: float
    requested_lon: float
    grid_lat: float | None = None
    grid_lon: float | None = None
    elevation_m: float | None = None
    days: list[DailyWeather] = field(default_factory=list)
    status: str = "nodata"  # ok, partial, nodata
    message: str = ""
    source: str = SOURCE
    retrieved_at_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _num(v):
    """float, or None if missing/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _http_get_json(url: str) -> dict:
    last = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise WeatherError(
                    f"Weather provider rejected the request (HTTP {exc.code}).") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError) as exc:
            last = exc
        logger.warning("Weather attempt %d/%d failed: %s", attempt, _MAX_ATTEMPTS, last)
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_RETRY_PAUSE_S)
    raise WeatherUnavailableError(
        f"Weather provider unavailable after {_MAX_ATTEMPTS} attempts: {last}")


def _validate(lat, lon, days):
    for name, v, lo, hi in (("lat", lat, -90, 90), ("lon", lon, -180, 180)):
        f = _num(v)
        if f is None or not lo <= f <= hi:
            raise InvalidWeatherRequestError(f"{name} must be between {lo} and {hi}.")
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= MAX_DAYS:
        raise InvalidWeatherRequestError(f"days must be an integer from 1 to {MAX_DAYS}.")
    return float(lat), float(lon)


def fetch_recent_weather(lat: float, lon: float, days: int = 7,
                         today: date | None = None) -> WeatherResult:
    """Daily max temperature, rainfall and ET0 for the last `days` completed days."""
    lat, lon = _validate(lat, lon, days)
    if today is None:
        today = datetime.now(ZoneInfo(TIMEZONE)).date()

    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "past_days": days, "forecast_days": 1, "timezone": TIMEZONE,
    })
    payload = _http_get_json(f"{BASE_URL}?{query}")

    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
        raise WeatherUnavailableError("Unexpected response from weather provider.")

    def pick(name, i):
        series = daily.get(name)
        if isinstance(series, list) and i < len(series):
            return _num(series[i])
        return None

    rows = []
    for i, d in enumerate(daily["time"]):
        if str(d) >= today.isoformat():  # exclude today's partial day and later
            continue
        rows.append(DailyWeather(
            date=str(d),
            max_temperature_c=pick("temperature_2m_max", i),
            precipitation_mm=pick("precipitation_sum", i),
            et0_mm=pick("et0_fao_evapotranspiration", i)))

    result = WeatherResult(
        requested_lat=lat, requested_lon=lon,
        grid_lat=_num(payload.get("latitude")),
        grid_lon=_num(payload.get("longitude")),
        elevation_m=_num(payload.get("elevation")),
        days=rows,
        retrieved_at_utc=datetime.now(timezone.utc).isoformat())

    if not rows:
        result.status, result.message = "nodata", "No completed days returned."
        return result

    attrs = ["max_temperature_c", "precipitation_mm", "et0_mm"]
    problems = []
    for attr in attrs:
        missing = sum(1 for r in rows if getattr(r, attr) is None)
        if missing:
            problems.append(f"{attr}: {missing} of {len(rows)} days missing")
    all_null = all(getattr(r, a) is None for r in rows for a in attrs)
    if all_null:
        result.status = "nodata"
    else:
        result.status = "partial" if problems else "ok"
    result.message = "; ".join(problems)
    return result

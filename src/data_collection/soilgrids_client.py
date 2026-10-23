import csv
import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.errors import RasterioIOError

logger = logging.getLogger(__name__)

BASE_URL = "/vsicurl/https://files.isric.org/soilgrids/latest/data"
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


@dataclass(frozen=True)
class PropertySpec:
    d_factor: float   # stored integer / d_factor = value in `unit`
    unit: str


# VERIFY these factors/units against the SoilGrids docs before relying on them.
PROPERTIES = {
    "sand":     PropertySpec(10, "%"),
    "silt":     PropertySpec(10, "%"),
    "clay":     PropertySpec(10, "%"),
    "phh2o":    PropertySpec(10, "pH (H2O)"),
    "soc":      PropertySpec(10, "g/kg"),
    "nitrogen": PropertySpec(100, "g/kg"),
    "cec":      PropertySpec(10, "cmol(c)/kg"),
    "bdod":     PropertySpec(100, "kg/dm3"),
    "cfvo":     PropertySpec(10, "%"),
}
DEFAULT_PROPERTIES = ["sand", "silt", "clay", "phh2o", "soc"]


class SoilGridsError(Exception):
    pass


class InvalidCoordinateError(SoilGridsError, ValueError):
    pass


class SoilGridsUnavailableError(SoilGridsError):
    pass


@dataclass
class Measurement:
    property: str
    depth: str
    statistic: str
    raw_value: float | None
    value: float | None        # None means missing, never 0
    unit: str
    status: str                # ok | nodata | outside_coverage | error
    message: str = ""
    read_via: str = ""         


@dataclass
class SoilResult:
    latitude: float
    longitude: float
    source: str
    retrieved_at_utc: str
    resolution_m: int
    measurements: list[Measurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_coordinates(lat: float, lon: float) -> tuple[float, float]:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinateError("Latitude/longitude must be numbers.") from exc
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise InvalidCoordinateError("Coordinates must be finite.")
    if not -90 <= lat <= 90:
        raise InvalidCoordinateError(f"Latitude {lat} outside [-90, 90].")
    if not -180 <= lon <= 180:
        raise InvalidCoordinateError(f"Longitude {lon} outside [-180, 180].")
    return lat, lon


def scale_value(raw, nodata, d_factor):
    """Return scaled value, or None if raw is missing."""
    if raw is None:
        return None
    raw = float(raw)
    if nodata is not None and raw == nodata:
        return None
    return raw / d_factor


def _layer_url(prop: str, depth: str, stat: str) -> str:
    return f"{BASE_URL}/{prop}/{prop}_{depth}_{stat}.vrt"


_GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".vrt,.tif,.tiff",
    GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_TIMEOUT="300",
)

# Retry: GDAL does not retry a dropped transfer, so we retry each layer here.
_MAX_ATTEMPTS = 3
_RETRY_PAUSE_S = 3

# Per-layer cache. Only permanent outcomes are cached; errors never are.
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "soilgrids"
CACHE_TTL_DAYS = 30
_CACHEABLE = {"ok", "nodata", "outside_coverage"}


def _cache_path(prop, depth, stat, lat, lon) -> Path:
    return CACHE_DIR / f"{prop}_{depth}_{stat}_{lat:.3f}_{lon:.3f}.json"


def _cache_get(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["cached_at_utc"])
        if datetime.now(timezone.utc) - cached_at > timedelta(days=CACHE_TTL_DAYS):
            return None
        return Measurement(**data["measurement"])
    except (OSError, ValueError, KeyError, TypeError):
        return None          # missing, corrupt or expired: just refetch


def _cache_put(path: Path, m: Measurement) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement": asdict(m),
        }), encoding="utf-8")
        tmp.replace(path)    # atomic, safe with parallel workers
    except OSError as exc:
        logger.warning("Could not write soil cache %s: %s", path, exc)


def _sample_remote(prop, depth, stat, lat, lon) -> Measurement:
    """One attempt at one layer. Raises RasterioIOError on a failed read."""
    spec = PROPERTIES[prop]
    m = Measurement(prop, depth, stat, None, None, spec.unit, "error", read_via="remote")
    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(_layer_url(prop, depth, stat)) as ds:
            x, y = Transformer.from_crs(
                "EPSG:4326", ds.crs, always_xy=True).transform(lon, lat)
            b = ds.bounds
            if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                m.status = "outside_coverage"
            else:
                raw = next(ds.sample([(x, y)]))[0]
                m.raw_value = float(raw)
                m.value = scale_value(raw, ds.nodata, spec.d_factor)
                m.status = "ok" if m.value is not None else "nodata"
    return m


LOCAL_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "soilgrids_tn"


def _sample_local(prop, depth, stat, lat, lon):
    """Read from the local Tamil Nadu window. None = no file or point outside window."""
    path = LOCAL_DIR / f"{prop}_{depth}_{stat}.tif"
    if not path.exists():
        return None
    spec = PROPERTIES[prop]
    with rasterio.open(path) as ds:
        x, y = Transformer.from_crs(
            "EPSG:4326", ds.crs, always_xy=True).transform(lon, lat)
        b = ds.bounds
        if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
            return None
        raw = next(ds.sample([(x, y)]))[0]
        value = scale_value(raw, ds.nodata, spec.d_factor)
    return Measurement(prop, depth, stat, float(raw), value, spec.unit,
                       "ok" if value is not None else "nodata", read_via="local")

def _sample_once(prop, depth, stat, lat, lon) -> Measurement:
    """Local window first; remote read only if the point is outside it."""
    m = _sample_local(prop, depth, stat, lat, lon)
    return m if m is not None else _sample_remote(prop, depth, stat, lat, lon)


def _read_layer(prop, depth, stat, lat, lon, use_cache=True):
    spec = PROPERTIES[prop]
    path = None
    if use_cache:
        # Round so nearby requests share an entry (3 decimals is about 110 m).
        lat, lon = round(lat, 3), round(lon, 3)
        path = _cache_path(prop, depth, stat, lat, lon)
        cached = _cache_get(path)
        if cached is not None:
            logger.info("%s %s -> %s %s (cache)", prop, depth, cached.value, cached.status)
            cached.read_via = "cache"
            return cached

    m = Measurement(prop, depth, stat, None, None, spec.unit, "error")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            m = _sample_once(prop, depth, stat, lat, lon)
            break
        except RasterioIOError as exc:
            m.message = f"Access error: {exc}"
            logger.warning("%s %s attempt %d/%d failed: %s",
                           prop, depth, attempt, _MAX_ATTEMPTS, exc)
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_PAUSE_S * attempt)
        except Exception as exc:
            m.message = f"Unexpected error: {exc}"
            logger.exception("%s %s failed", prop, depth)
            break

    if path is not None and m.status in _CACHEABLE:
        _cache_put(path, m)
    logger.info("%s %s -> %s %s", prop, depth, m.value, m.status)
    return m


def fetch_soil_properties(
    lat: float,
    lon: float,
    properties: list[str] | None = None,
    depths: list[str] | None = None,
    stat: str = "mean",
    max_workers: int = 2,
    use_cache: bool = True,
) -> SoilResult:
    lat, lon = validate_coordinates(lat, lon)
    properties = properties or DEFAULT_PROPERTIES
    depths = depths or DEPTHS
    unknown = [p for p in properties if p not in PROPERTIES]
    if unknown:
        raise ValueError(f"Unsupported properties: {unknown}")
    bad_depths = [d for d in depths if d not in DEPTHS]
    if bad_depths:
        raise ValueError(f"Unsupported depths: {bad_depths}")

    result = SoilResult(
        latitude=lat, longitude=lon,
        source="ISRIC SoilGrids 2.0 via files.isric.org (VRT, CC BY 4.0)",
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        resolution_m=250,
    )
    jobs = [(p, d) for p in properties for d in depths]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        result.measurements = list(
            ex.map(lambda pd: _read_layer(pd[0], pd[1], stat, lat, lon, use_cache), jobs))

    if all(m.status == "error" for m in result.measurements):
        raise SoilGridsUnavailableError(
            "No layer could be read. Check internet access, GDAL/rasterio install, "
            "or whether files.isric.org is reachable.")
    return result


def save_json(result: SoilResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def save_csv(result: SoilResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["latitude", "longitude", "retrieved_at_utc", "source", "property",
                     "depth", "statistic", "raw_value", "value", "unit", "status", "message", "read_via"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in result.measurements:
            w.writerow({"latitude": result.latitude, "longitude": result.longitude,
                        "retrieved_at_utc": result.retrieved_at_utc,
                        "source": result.source, **asdict(m)})

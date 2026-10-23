"""POST /assess: rule-engine output for a crop and location.

Preliminary rule-based decision support, not a validated model prediction.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agronomy.crop_config import load_config
from src.agronomy.risk_rules import assess_all
from src.data_collection.soilgrids_client import (
    fetch_soil_properties, SoilGridsError)
from src.data_collection.weather_client import (
    fetch_recent_weather, WeatherError)
from src.preprocessing.soil_preprocessor import preprocess_record

logger = logging.getLogger(__name__)
router = APIRouter()
CONFIG = load_config()


class Observations(BaseModel):
    max_temperature_c: float | None = None
    standing_water_days: float | None = Field(None, ge=0)


class AssessRequest(BaseModel):
    crop: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    growth_stage: str | None = None
    observations: Observations = Field(default_factory=Observations)
    include_soil: bool = True
    include_weather: bool = True


class InputUsed(BaseModel):
    value: float | str | bool | None = None
    provenance: str  # seen: "manual", "retrieved"


class Assessment(BaseModel):
    category: Literal["waterlogging", "heat_stress", "water_stress",
                      "disease", "soil_suitability"]
    status: Literal["assessed", "incomplete"]
    level: Literal["elevated", "not_elevated",
                   "within_documented_ph_range",
                   "outside_documented_ph_range"] | None = None
    reasons: list[str] = []
    missing_inputs: list[str] = []
    inputs_used: dict[str, InputUsed] = {}
    references: list[str] = []
    caveats: list[str] = []
    basis: str = "preliminary_rule_based"


class Location(BaseModel):
    lat: float
    lon: float


class SoilSummary(BaseModel):
    status: Literal["ok", "partial", "nodata", "unavailable", "not_requested"]
    message: str = ""  # names failing layers; empty when ok
    layers_ok: int = 0
    read_via: list[str] = []  # local, remote, cache


class WeatherSummary(BaseModel):
    status: Literal["ok", "partial", "nodata", "unavailable", "not_requested"]
    message: str = ""  # names gaps or the failure; empty when ok
    days_ok: int = 0  # completed days with a max temperature
    window_days: int = 0
    hottest_day: str | None = None
    max_temperature_c: float | None = None  # highest daily max in window
    grid_lat: float | None = None
    grid_lon: float | None = None
    elevation_m: float | None = None
    source: str = ""
    retrieved_at_utc: str | None = None


class AssessResponse(BaseModel):
    crop: str
    growth_stage: str | None = None
    growth_stage_note: str | None = None
    disclaimer: str
    assessments: list[Assessment]
    location: Location
    soil: SoilSummary
    weather: WeatherSummary


@router.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest):
    # Plain def: blocking network I/O runs in a worker thread.
    soil: dict = {}
    soil_info = {"status": "not_requested", "message": "", "layers_ok": 0,
                 "read_via": []}

    if req.include_soil:
        try:
            rec = fetch_soil_properties(
                req.lat, req.lon, ["sand", "silt", "clay", "phh2o"], ["0-5cm"]
            ).to_dict()
            soil = preprocess_record(rec)
            ok = sum(1 for m in rec["measurements"] if m["status"] == "ok")
            soil_info["layers_ok"] = ok
            soil_info["status"] = "ok" if ok == 4 else ("partial" if ok else "nodata")
            problems = []
            for m in rec["measurements"]:
                if m["status"] != "ok":
                    problems.append(
                        f'{m["property"]}_{m["depth"]}: {m["message"] or m["status"]}')
            soil_info["message"] = "; ".join(problems)
            soil_info["read_via"] = sorted(
                {m["read_via"] for m in rec["measurements"] if m["read_via"]})
        except SoilGridsError as e:
            logger.warning("Soil fetch failed: %s", e)
            soil_info.update(status="unavailable", message=str(e))

    weather_info = {"status": "not_requested", "message": ""}
    weather_input = None

    if req.include_weather:
        try:
            w = fetch_recent_weather(req.lat, req.lon, days=7)
            with_t = [d for d in w.days if d.max_temperature_c is not None]
            weather_info = {
                "status": w.status, "message": w.message,
                "days_ok": len(with_t), "window_days": len(w.days),
                "grid_lat": w.grid_lat, "grid_lon": w.grid_lon,
                "elevation_m": w.elevation_m, "source": w.source,
                "retrieved_at_utc": w.retrieved_at_utc,
            }
            if with_t:
                hot = max(with_t, key=lambda d: d.max_temperature_c)
                weather_info["hottest_day"] = hot.date
                weather_info["max_temperature_c"] = hot.max_temperature_c
                weather_input = {"max_temperature_c": hot.max_temperature_c,
                                 "date": hot.date, "window_days": len(w.days)}
        except WeatherError as e:
            logger.warning("Weather fetch failed: %s", e)
            weather_info = {"status": "unavailable", "message": str(e)}

    try:
        result = assess_all(
            CONFIG, req.crop, req.growth_stage, soil,
            req.observations.model_dump(), weather=weather_input)
    except KeyError as e:
        raise HTTPException(422, str(e.args[0]))
    except ValueError as e:
        raise HTTPException(422, str(e))

    result["location"] = {"lat": req.lat, "lon": req.lon}
    result["soil"] = soil_info
    result["weather"] = weather_info
    return result

"""POST /assess: rule-engine output for a crop and location.

Preliminary rule-based decision support, not a validated model prediction.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agronomy.crop_config import load_config
from src.agronomy.risk_rules import assess_all
from src.data_collection.soilgrids_client import (
    fetch_soil_properties, SoilGridsError)
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


@router.post("/assess")
def assess(req: AssessRequest):
    # Plain def: blocking network I/O runs in a worker thread.
    soil: dict = {}
    soil_info = {"status": "not_requested", "message": "", "layers_ok": 0}

    if req.include_soil:
        try:
            rec = fetch_soil_properties(
                req.lat, req.lon, ["sand", "silt", "clay", "phh2o"], ["0-5cm"]
            ).to_dict()
            soil = preprocess_record(rec)
            ok = sum(1 for m in rec["measurements"] if m["status"] == "ok")
            soil_info["layers_ok"] = ok
            soil_info["status"] = "ok" if ok == 4 else ("partial" if ok else "nodata")
        except SoilGridsError as e:
            logger.warning("Soil fetch failed: %s", e)
            soil_info.update(status="unavailable", message=str(e))

    try:
        result = assess_all(
            CONFIG, req.crop, req.growth_stage, soil,
            req.observations.model_dump())
    except KeyError as e:
        raise HTTPException(422, str(e.args[0]))
    except ValueError as e:
        raise HTTPException(422, str(e))

    result["location"] = {"lat": req.lat, "lon": req.lon}
    result["soil"] = soil_info
    return result

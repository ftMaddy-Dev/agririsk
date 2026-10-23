import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from src.data_collection.soilgrids_client import (
    fetch_soil_properties, PROPERTIES, DEPTHS, DEFAULT_PROPERTIES,
    InvalidCoordinateError, SoilGridsUnavailableError,
)

logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="AgriRisk API", version="0.1.0",
    description="Soil data retrieval only. Risk, explanation and advisory endpoints are not implemented yet.")


class MeasurementOut(BaseModel):
    property: str
    depth: str
    statistic: str
    raw_value: float | None
    value: float | None
    unit: str
    status: str
    message: str


class SoilOut(BaseModel):
    latitude: float
    longitude: float
    source: str
    retrieved_at_utc: str
    resolution_m: int
    measurements: list[MeasurementOut]


@app.get("/soil", response_model=SoilOut)
def get_soil(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    properties: list[str] = Query(DEFAULT_PROPERTIES),
    depths: list[str] = Query(DEPTHS),
):
    bad_p = [p for p in properties if p not in PROPERTIES]
    bad_d = [d for d in depths if d not in DEPTHS]
    if bad_p or bad_d:
        raise HTTPException(422, f"Unsupported properties {bad_p} or depths {bad_d}")
    try:
        return fetch_soil_properties(lat, lon, properties, depths).to_dict()
    except InvalidCoordinateError as e:
        raise HTTPException(422, str(e))
    except SoilGridsUnavailableError as e:
        raise HTTPException(503, str(e))

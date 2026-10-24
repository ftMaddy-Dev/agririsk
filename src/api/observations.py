"""POST/GET /observations: log and list field reports."""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query

from src.observations import store
from src.observations.schema import ObservationIn, ObservationOut

router = APIRouter()
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "observations.db"


@router.post("/observations", response_model=ObservationOut, status_code=201)
def create_observation(obs: ObservationIn):
    return store.add(DB_PATH, obs)


@router.get("/observations", response_model=list[ObservationOut])
def get_observations(
    crop: Literal["sesame", "groundnut"] | None = None,
    problem: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    return store.list_observations(DB_PATH, crop, problem, limit)

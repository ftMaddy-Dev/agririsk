"""GET /models, POST /predict-risk, /explain-risk, /generate-advisory (groundnut surrogate models)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agronomy.advisory_rules import build_advisory
from src.explainability.shap_explainer import explain_one
from src.models import predict as pm

router = APIRouter()


class RiskRequest(BaseModel):
    crop: Literal["sesame", "groundnut"] = "groundnut"
    model: str = Field(..., description="Saved model name, see GET /models")
    features: dict[str, float | None] = Field(..., min_length=1,
                                              description="Feature values; omitted or null means missing")


class ExplainRequest(RiskRequest):
    top_k: int = Field(5, ge=1, le=20)


class AdvisoryRequest(ExplainRequest):
    growth_stage: str | None = None


def _bundle(req: RiskRequest) -> dict:
    if req.crop != "groundnut":
        raise HTTPException(422, "Models are trained for groundnut only; no sesame model exists.")
    try:
        return pm.load_bundle(req.model, pm.MODEL_DIR)
    except pm.ModelNotFoundError as exc:
        raise HTTPException(404, str(exc))


def _predict(bundle: dict, features: dict) -> dict:
    try:
        return pm.predict_one(bundle, features)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


def _explain(bundle: dict, features: dict, top_k: int) -> dict:
    try:
        return explain_one(bundle, features, top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/models")
def models():
    return {"models": pm.list_models(pm.MODEL_DIR), "warning": pm.WARNING}


@router.post("/predict-risk")
def predict_risk(req: RiskRequest):
    return {"crop": req.crop, "model": req.model, "basis": "surrogate_model",
            **_predict(_bundle(req), req.features)}


@router.post("/explain-risk")
def explain_risk(req: ExplainRequest):
    b = _bundle(req)
    return {"crop": req.crop, "model": req.model, "basis": "surrogate_model",
            "prediction": _predict(b, req.features),
            "explanation": _explain(b, req.features, req.top_k)}


@router.post("/generate-advisory")
def generate_advisory(req: AdvisoryRequest):
    b = _bundle(req)
    return build_advisory(req.crop, req.growth_stage, _predict(b, req.features),
                          _explain(b, req.features, req.top_k))

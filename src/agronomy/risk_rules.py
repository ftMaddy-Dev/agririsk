"""Preliminary rule-based risk assessment.

NOT a validated ML prediction. Each category returns status 'assessed' or
'incomplete'; incomplete results list exactly which inputs are missing.
Thresholds come only from crop_config.json (verified entries).
"""
from __future__ import annotations

import math

from src.agronomy.crop_config import get_crop, get_parameter, usable_value

DISCLAIMER = "Preliminary rule-based decision support; not a validated model prediction."
CATEGORIES = ["waterlogging", "heat_stress", "water_stress", "disease", "soil_suitability"]


def _num(v):
    """float, or None if missing/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _usable(config, crop, name):
    try:
        return usable_value(config, crop, name)
    except KeyError:
        return None


def _source(config, crop, name):
    try:
        return get_parameter(config, crop, name).get("source", "")
    except KeyError:
        return ""


def _result(category, status, level=None, reasons=None, missing=None,
            inputs_used=None, references=None, caveats=None):
    return {
        "category": category,
        "status": status,                 # assessed | incomplete
        "level": level,                   # None unless assessed
        "reasons": reasons or [],
        "missing_inputs": missing or [],
        "inputs_used": inputs_used or {},
        "references": [r for r in (references or []) if r],
        "caveats": caveats or [],
        "basis": "preliminary_rule_based",
    }


def _check_obs(obs, key, minimum=None):
    if key not in obs or obs[key] is None:
        return None
    v = _num(obs[key])
    if v is None:
        raise ValueError(f"{key} must be a number")
    if minimum is not None and v < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return v


def assess_waterlogging(config, crop, stage, soil, obs):
    tol = _usable(config, crop, "waterlogging_tolerance_days")
    days = _check_obs(obs, "standing_water_days", 0)
    missing = []
    if tol is None:
        missing.append("verified waterlogging tolerance (days) for this crop")
    if days is None:
        missing.append("observed standing-water duration in days (manual or measured)")
    if missing:
        return _result("waterlogging", "incomplete", missing=missing,
                       reasons=["Soil texture alone is not used to infer waterlogging."])
    note = get_parameter(config, crop, "waterlogging_tolerance_days").get("note", "")
    level = "elevated" if days > tol else "not_elevated"
    return _result(
        "waterlogging", "assessed", level=level,
        reasons=[f"Observed standing water {days} d vs documented tolerance {tol} d."],
        inputs_used={"standing_water_days": {"value": days, "provenance": "manual"}},
        references=[_source(config, crop, "waterlogging_tolerance_days")],
        caveats=[note, "Qualitative documented tolerance, not a validated damage threshold."])


def assess_heat_stress(config, crop, stage, soil, obs, weather=None):
    tmax = _usable(config, crop, "temperature_max_c")
    t = _check_obs(obs, "max_temperature_c")
    provenance = "manual"
    if t is None and weather:
        wt = _num(weather.get("max_temperature_c"))
        if wt is not None:
            t, provenance = wt, "retrieved"
    missing = []
    if tmax is None:
        missing.append("verified T_Max for this crop")
    if t is None:
        missing.append("observed or retrieved maximum temperature (manual entry allowed)")
    if missing:
        return _result("heat_stress", "incomplete", missing=missing)
    note = get_parameter(config, crop, "temperature_max_c").get("note", "")
    level = "elevated" if t > tmax else "not_elevated"
    caveats = [note, "Growth-stage-specific heat thresholds are not available."]
    if provenance == "retrieved":
        n = weather.get("window_days")
        reasons = [f"Highest daily max in the last {n} completed days was {t} C "
                   f"on {weather.get('date')} vs documented T_Max {tmax} C."]
        caveats += ["Using the highest daily max in the window is a design choice, not a TNAU rule.",
                    "Weather is a modelled grid estimate, not a station reading."]
    else:
        reasons = [f"Max temperature {t} C vs documented T_Max {tmax} C."]
    return _result(
        "heat_stress", "assessed", level=level,
        reasons=reasons,
        inputs_used={"max_temperature_c": {"value": t, "provenance": provenance}},
        references=[_source(config, crop, "temperature_max_c")],
        caveats=caveats)


def assess_water_stress(config, crop, stage, soil, obs):
    return _result(
        "water_stress", "incomplete",
        missing=["rainfall and evaporation, or soil-moisture observations (weather provider on hold)",
                 "verified numeric water-stress thresholds for this crop and stage"],
        reasons=["No verified numeric water-stress threshold is configured. "
                 "IW/CPE irrigation ratios (where documented) need pan evaporation data."])


def assess_disease(config, crop, stage, soil, obs):
    return _result(
        "disease", "incomplete",
        missing=["verified disease-conducive weather thresholds",
                 "field scouting observations"],
        reasons=["Disease cannot be diagnosed from weather and soil properties alone."])


def assess_soil_suitability(config, crop, stage, soil, obs, ph_depth=None):
    ph_range = _usable(config, crop, "soil_ph_range")
    context = {}
    for key in ("phh2o_0_5cm_mean", "clay_0_5cm_mean", "sand_0_5cm_mean", "silt_0_5cm_mean"):
        v = _num(soil.get(key))
        if v is not None:
            context[key] = {"value": v, "provenance": "retrieved"}
    missing = []
    if ph_range is None:
        missing.append("verified soil pH range for this crop (TNAU)")
    if ph_depth is None:
        missing.append("agronomic root-zone depth to evaluate (not defined)")
    ph = None
    if not missing:
        ph = _num(soil.get(f"phh2o_{ph_depth}_mean"))
        if ph is None:
            missing.append(f"measured pH at depth {ph_depth}")
    if missing:
        return _result("soil_suitability", "incomplete", missing=missing,
                       inputs_used=context,
                       reasons=["Measured soil values are shown for context only; no suitability threshold applied."])
    lo, hi = ph_range
    level = "within_documented_ph_range" if lo <= ph <= hi else "outside_documented_ph_range"
    return _result(
        "soil_suitability", "assessed", level=level,
        reasons=[f"pH {ph} at depth {ph_depth} vs documented range {lo}-{hi}."],
        inputs_used={f"phh2o_{ph_depth}_mean": {"value": ph, "provenance": "retrieved"}},
        references=[_source(config, crop, "soil_ph_range")],
        caveats=["pH only; texture, drainage and nutrients not evaluated.",
                 "SoilGrids is a 250 m modelled estimate, not a field measurement."])


def assess_all(config: dict, crop: str, growth_stage: str | None = None,
               soil: dict | None = None, observations: dict | None = None,
               ph_depth: str | None = None, weather: dict | None = None) -> dict:
    get_crop(config, crop)  # raises KeyError for unknown crop
    soil, obs = soil or {}, observations or {}
    return {
        "crop": crop.lower(),
        "growth_stage": growth_stage,
        "growth_stage_note": "Stage recorded but not used: stage-wise thresholds are unresolved.",
        "disclaimer": DISCLAIMER,
        "assessments": [
            assess_waterlogging(config, crop, growth_stage, soil, obs),
            assess_heat_stress(config, crop, growth_stage, soil, obs, weather),
            assess_water_stress(config, crop, growth_stage, soil, obs),
            assess_disease(config, crop, growth_stage, soil, obs),
            assess_soil_suitability(config, crop, growth_stage, soil, obs, ph_depth),
        ],
    }

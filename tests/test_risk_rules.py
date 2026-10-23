import copy
import pytest
from src.agronomy.crop_config import load_config
from src.agronomy.risk_rules import assess_all, assess_soil_suitability, CATEGORIES

cfg = load_config()


def get(res, cat):
    return next(a for a in res["assessments"] if a["category"] == cat)


def test_all_categories_and_disclaimer():
    r = assess_all(cfg, "groundnut")
    assert [a["category"] for a in r["assessments"]] == CATEGORIES
    assert "not a validated" in r["disclaimer"]


def test_water_stress_and_disease_always_incomplete():
    r = assess_all(cfg, "sesame", observations={"max_temperature_c": 45, "standing_water_days": 30})
    for cat in ("water_stress", "disease"):
        a = get(r, cat)
        assert a["status"] == "incomplete" and a["level"] is None and a["missing_inputs"]


def test_waterlogging_not_inferred_from_texture():
    r = assess_all(cfg, "groundnut", soil={"clay_0_5cm_mean": 60.0})
    a = get(r, "waterlogging")
    assert a["status"] == "incomplete"
    assert any("texture" in s for s in a["reasons"])


def test_waterlogging_groundnut_with_observation():
    assert get(assess_all(cfg, "groundnut", observations={"standing_water_days": 10}),
               "waterlogging")["level"] == "elevated"
    assert get(assess_all(cfg, "groundnut", observations={"standing_water_days": 3}),
               "waterlogging")["level"] == "not_elevated"


def test_waterlogging_sesame_incomplete_tolerance_unresolved():
    a = get(assess_all(cfg, "sesame", observations={"standing_water_days": 10}), "waterlogging")
    assert a["status"] == "incomplete"


def test_heat_stress():
    assert get(assess_all(cfg, "groundnut", observations={"max_temperature_c": 42}),
               "heat_stress")["level"] == "elevated"
    assert get(assess_all(cfg, "sesame", observations={"max_temperature_c": 35}),
               "heat_stress")["level"] == "not_elevated"
    assert get(assess_all(cfg, "sesame"), "heat_stress")["status"] == "incomplete"


def test_soil_suitability_incomplete_with_real_config():
    a = get(assess_all(cfg, "sesame", soil={"phh2o_0_5cm_mean": 7.0}), "soil_suitability")
    assert a["status"] == "incomplete"
    assert "phh2o_0_5cm_mean" in a["inputs_used"]


def test_soil_suitability_assessed_when_range_and_depth_exist():
    fake = copy.deepcopy(cfg)
    fake["crops"]["sesame"]["parameters"]["soil_ph_range"].update(value=[6.0, 7.5], status="verified")
    a = assess_soil_suitability(fake, "sesame", None, {"phh2o_0_5cm_mean": 7.0}, {}, ph_depth="0_5cm")
    assert a["status"] == "assessed" and a["level"] == "within_documented_ph_range"


def test_invalid_observation_and_unknown_crop():
    with pytest.raises(ValueError):
        assess_all(cfg, "groundnut", observations={"standing_water_days": -1})
    with pytest.raises(KeyError):
        assess_all(cfg, "wheat")

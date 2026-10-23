import pytest
from src.agronomy.crop_config import (load_config, usable_value, unresolved,
                                      get_crop)

cfg = load_config()


def test_verified_value_returned():
    assert usable_value(cfg, "sesame", "temperature_optimum_c") == [25, 35]


def test_unresolved_never_returned():
    assert usable_value(cfg, "sesame", "soil_ph_range") is None
    assert usable_value(cfg, "groundnut", "soil_ph_range") is None
    assert "soil_ph_range" in unresolved(cfg, "sesame")


def test_unknown_crop():
    with pytest.raises(KeyError):
        get_crop(cfg, "wheat")


def test_verified_entries_have_sources():
    for crop in cfg["crops"].values():
        for name, p in crop["parameters"].items():
            if p["status"] in ("verified", "verified_excerpt"):
                assert p["source"].startswith("https://agritech.tnau.ac.in"), name


def test_groundnut_flooding_tolerance_from_cpg():
    assert usable_value(cfg, "groundnut", "waterlogging_tolerance_days") == 7


def test_stage_markers_are_partial_not_complete():
    for crop in cfg["crops"].values():
        assert crop["growth_stages"]["status"] == "verified_partial"
        assert "unresolved" in crop["growth_stages"]

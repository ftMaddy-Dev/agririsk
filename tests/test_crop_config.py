import pytest
from src.agronomy.crop_config import (load_config, usable_value, unresolved,
                                      get_crop, source_problem, SOURCE_HOSTS)

cfg = load_config()


def test_verified_value_returned():
    assert usable_value(cfg, "sesame", "temperature_optimum_c") == [25, 35]


def test_unresolved_never_returned():
    # No numeric waterlogging tolerance exists in TNAU or the ICAR sesame guide.
    assert usable_value(cfg, "sesame", "waterlogging_tolerance_days") is None
    assert usable_value(cfg, "groundnut", "soil_ph_range") is None
    assert "waterlogging_tolerance_days" in unresolved(cfg, "sesame")


def test_icar_sesame_ph_is_returned():
    assert usable_value(cfg, "sesame", "soil_ph_range") == [5.5, 8.0]
    assert "soil_ph_range" not in unresolved(cfg, "sesame")


def test_unknown_crop():
    with pytest.raises(KeyError):
        get_crop(cfg, "wheat")


def test_verified_entries_have_sources():
    for crop in cfg["crops"].values():
        for name, p in crop["parameters"].items():
            if p["status"] in ("verified", "verified_excerpt"):
                assert p.get("source_type") in SOURCE_HOSTS, name
                assert source_problem(p) is None, (name, source_problem(p))


def test_source_type_must_match_host():
    assert source_problem({"source": "https://agritech.tnau.ac.in/x",
                           "source_type": "tnau"}) is None
    assert source_problem({"source": "https://example.com/x",
                           "source_type": "tnau"}) is not None
    assert source_problem({"source": "https://agritech.tnau.ac.in/x",
                           "source_type": "icar"}) is not None
    assert source_problem({"source": "https://agritech.tnau.ac.in/x"}) is not None


def test_icar_host_accepted_and_lookalike_rejected():
    assert source_problem({"source": "https://www.icar.gov.in/x",
                           "source_type": "icar"}) is None
    assert source_problem({"source": "https://icar.gov.in.evil.com/x",
                           "source_type": "icar"}) is not None


def test_groundnut_flooding_tolerance_from_cpg():
    assert usable_value(cfg, "groundnut", "waterlogging_tolerance_days") == 7


def test_stage_markers_are_partial_not_complete():
    for crop in cfg["crops"].values():
        assert crop["growth_stages"]["status"] == "verified_partial"
        assert "unresolved" in crop["growth_stages"]

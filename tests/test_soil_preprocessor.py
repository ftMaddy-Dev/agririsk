import math
from src.preprocessing.soil_preprocessor import preprocess_record


def _rec(sand, silt, clay):
    ms = []
    for p, v in (("sand", sand), ("silt", silt), ("clay", clay)):
        ms.append({"property": p, "depth": "0-5cm", "statistic": "mean",
                   "raw_value": None, "value": v, "unit": "%",
                   "status": "ok" if v is not None else "nodata", "message": ""})
    return {"latitude": 11.99, "longitude": 79.49, "source": "t",
            "retrieved_at_utc": "t", "resolution_m": 250, "measurements": ms}


def test_missing_preserved_not_zero():
    row = preprocess_record(_rec(40, None, 32))
    assert math.isnan(row["silt_0_5cm_mean"])
    assert row["silt_0_5cm_mean_status"] == "nodata"
    assert row["texture_sum_0_5cm_ok"] is None


def test_out_of_range_flagged():
    row = preprocess_record(_rec(140, 27, 32))
    assert math.isnan(row["sand_0_5cm_mean"])
    assert row["sand_0_5cm_mean_status"] == "out_of_range"


def test_texture_sum():
    assert preprocess_record(_rec(39.9, 27.5, 32.7))["texture_sum_0_5cm_ok"] is True
    assert preprocess_record(_rec(40, 40, 40))["texture_sum_0_5cm_ok"] is False

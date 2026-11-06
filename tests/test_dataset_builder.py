import datetime as dt
import math
import pandas as pd
from src.models.dataset_builder import (build_dataset, validate_dataset, heat_label,
                                        WEATHER_FEATURES, LEAKY, LABEL_ORIGIN)

TODAY = dt.date(2026, 11, 6)
CAL = pd.DataFrame([{"crop": "groundnut", "district": "Villupuram", "season_name": "Chithiraipattam",
                     "regime": "irrigated", "sow_month_start": 4, "sow_month_end": 5, "source": "t"}])
CENT = pd.DataFrame([{"district": "Villupuram", "lat": 11.9, "lon": 79.5, "source": "t"}])
SOIL = lambda lat, lon: {"sand_0_5cm_mean": 40.0}


def weather(hot_years=(), none_years=()):
    def f(lat, lon, start, end):
        days, d = [], start
        while d <= end:
            if start.year in none_years:
                t = None
            else:
                t = 41.0 if (start.year in hot_years and d == start) else 30.0
            days.append({"date": d.isoformat(), "tmax_c": t, "tmin_c": 22.0,
                         "precip_mm": 2.0, "et0_mm": 4.0})
            d += dt.timedelta(days=1)
        return {"days": days}
    return f


def build(years, **kw):
    return build_dataset(CAL, CENT, years, 40, weather(**kw), SOIL, today=TODAY)


def test_labels_and_origin():
    df, _ = build([2015, 2019], hot_years={2019})
    assert df.sort_values("year_start")["label_heat_flag"].tolist() == [0, 1]
    assert (df["label_origin"] == LABEL_ORIGIN).all()


def test_missing_weather_gives_missing_label_not_zero():
    df, _ = build([2015], none_years={2015})
    assert df["label_heat_flag"].isna().all() and heat_label(float("nan")) is None


def test_future_window_skipped():
    df, skipped = build([2015, 2027])
    assert len(df) == 1 and any("2027" in s for s in skipped)


def test_no_label_leakage_in_feature_list():
    assert not (set(WEATHER_FEATURES) & LEAKY)


def test_validate_dataset():
    df, _ = build([2015, 2019], hot_years={2019})
    assert validate_dataset(df) == []
    bad = df.copy()
    bad.loc[bad.index[0], "label_heat_flag"] = 2
    assert validate_dataset(bad)
    assert any("duplicate" in p for p in validate_dataset(pd.concat([df, df])))

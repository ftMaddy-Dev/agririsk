import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.dataset_builder import LEAKY
from src.models.train import (prepare, feature_sets, evaluate_loyo, evaluate_logo, metrics)


def toy():
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2015, 2021):
        for dist in ("A", "B", "C"):
            hot = int(year in (2016, 2019) and dist != "C")
            rows.append({"district": dist, "year_start": year, "window_start": f"{year}-04-01",
                         "regime": "irrigated", "label_heat_flag": hot,
                         "tmin_mean_c": 22 + hot * 2 + rng.normal(0, .3), "rain_sum_mm": 300.0,
                         "rain_max_day_mm": 40.0, "et0_sum_mm": 500.0, "rain_minus_et0_mm": -200.0,
                         "sand_0_5cm_mean": 40.0, "silt_0_5cm_mean": 30.0,
                         "clay_0_5cm_mean": 30.0, "phh2o_0_5cm_mean": 7.0})
    rows.append({**rows[0], "year_start": 2021, "label_heat_flag": np.nan})
    return pd.DataFrame(rows)


def spy_factory(seen):
    class Spy(RandomForestClassifier):
        def fit(self, X, y, *a, **k):
            seen.append(len(X))
            return super().fit(X, y, *a, **k)
    return lambda: Spy(n_estimators=10, random_state=42)


def test_prepare_drops_missing_labels_never_zero_fills():
    d = prepare(toy())
    assert len(d) == 18 and d["label_heat_flag"].notna().all()


def test_feature_sets_exclude_label_features_and_differ():
    s = feature_sets(prepare(toy()))
    for cols in s.values():
        assert not (set(cols) & LEAKY)
    assert set(s["weather_only"]) < set(s["context_aware"])


def test_baselines_contain_only_season_or_district():
    s = feature_sets(prepare(toy()))
    assert s["season_only"] == ["window_start_month"]
    assert s["district_only"] and all(c.startswith("dist_") for c in s["district_only"])


def test_loyo_covers_every_row_and_holds_out_whole_years():
    d = prepare(toy())
    seen = []
    oof = evaluate_loyo(d, feature_sets(d)["weather_only"], spy_factory(seen))
    assert not np.isnan(oof).any() and len(oof) == len(d)
    assert set(seen) == {15}      # 18 rows minus one 3-district year


def test_logo_holds_out_whole_districts():
    d = prepare(toy())
    seen = []
    oof = evaluate_logo(d, feature_sets(d)["weather_only"], spy_factory(seen))
    assert not np.isnan(oof).any() and len(oof) == len(d)
    assert set(seen) == {12}      # 18 rows minus one 6-row district


def test_district_only_cannot_identify_unseen_district():
    d = prepare(toy())
    cols = feature_sets(d)["district_only"]
    held = d[d["district"] == "A"]
    assert held[cols].drop(columns=["dist_A"]).sum().sum() == 0   # unseen district looks like "all other"


def test_metrics_shape_and_single_class():
    m = metrics([0, 1, 1, 0], np.array([.1, .9, .6, .4]))
    assert m["roc_auc"] == 1.0 and set(m) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert np.isnan(metrics([0, 0], np.array([.1, .2]))["roc_auc"])

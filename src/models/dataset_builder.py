"""Season-level training dataset builder.

Label here is a RULE-DERIVED SURROGATE (label_origin), not an observed outcome.
Observed-yield labels can replace it by setting label_origin accordingly.
"""
from __future__ import annotations

import datetime as dt
import logging
import math

import pandas as pd

from src.preprocessing.season_features import compute_season_features, season_window

logger = logging.getLogger(__name__)

WINDOW_DAYS = 105   # assumption: CPG 2020 p.192 irrigated groundnut phase table spans 105 days
SOIL_PROPS = ["sand", "silt", "clay", "phh2o"]   # layers present in the local TN window
SOIL_DEPTHS = ["0-5cm"]   # assumption: top 5 cm only; not an agronomic root zone
LABEL_ORIGIN = "rule_derived_surrogate"
LABEL_RULE = "label_heat_flag = 1 if hot_days >= 1 (daily max above configured T_Max); assumption"
KEY = ["crop", "district", "season_name", "year_start"]

# Label is derived from daily maximum temperature, so these must not be features.
LEAKY = {"tmax_mean_c", "tmax_max_c", "hot_days", "tmax_coverage"}
WEATHER_FEATURES = ["tmin_mean_c", "rain_sum_mm", "rain_max_day_mm", "et0_sum_mm", "rain_minus_et0_mm"]
SOIL_FEATURES = [f"{p}_0_5cm_mean" for p in SOIL_PROPS]


def heat_label(hot_days):
    """1/0 from hot_days; None (never 0) when hot_days is unavailable."""
    if hot_days is None or (isinstance(hot_days, float) and math.isnan(hot_days)):
        return None
    return 1 if hot_days >= 1 else 0


def build_dataset(calendar, centroids, years, hot_threshold_c, weather_fetcher,
                  soil_fetcher, today=None, crop="groundnut"):
    """Return (DataFrame, skipped list of strings). Fetchers are injected for testing."""
    today = today or dt.date.today()
    cent = centroids.set_index("district")
    rows, skipped, soil_cache = [], [], {}
    for _, c in calendar[calendar["crop"] == crop].iterrows():
        d = c["district"]
        if d not in cent.index:
            skipped.append(f"{d}: no centroid")
            continue
        lat, lon = float(cent.loc[d, "lat"]), float(cent.loc[d, "lon"])
        if d not in soil_cache:
            try:
                soil_cache[d] = soil_fetcher(lat, lon)
            except Exception as exc:
                skipped.append(f"{d}: soil fetch failed ({exc}); soil features missing")
                soil_cache[d] = {}
        soil = {k: soil_cache[d].get(k, float("nan")) for k in SOIL_FEATURES}
        soil["soil_points_used"] = soil_cache[d].get("soil_points_used", 0)
        for y in years:
            start, end = season_window(y, f"{int(c['sow_month_start']):02d}-01", WINDOW_DAYS)
            tag = f"{d} {c['season_name']} {y}"
            if end > today - dt.timedelta(days=2):
                skipped.append(f"{tag}: window not completed")
                continue
            try:
                days = weather_fetcher(lat, lon, start, end)["days"]
            except Exception as exc:
                skipped.append(f"{tag}: weather fetch failed ({exc})")
                continue
            f = compute_season_features(days, start, end, hot_threshold_c)
            row = {"crop": crop, "district": d, "season_name": c["season_name"],
                   "regime": c["regime"], "year_start": y, "lat": lat, "lon": lon, **f, **soil}
            row["label_heat_flag"] = heat_label(f["hot_days"])
            row["label_origin"] = LABEL_ORIGIN
            row["label_rule"] = LABEL_RULE
            rows.append(row)
    return pd.DataFrame(rows), skipped


def validate_dataset(df):
    problems = []
    need = KEY + ["label_heat_flag", "label_origin"] + WEATHER_FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if not df["label_heat_flag"].dropna().isin([0, 1]).all():
        problems.append("label_heat_flag has values other than 0/1")
    if not df["label_origin"].isin([LABEL_ORIGIN, "observed_yield_derived"]).all():
        problems.append("unknown label_origin")
    if df.duplicated(KEY).any():
        problems.append(f"duplicate rows for {KEY}")
    return problems

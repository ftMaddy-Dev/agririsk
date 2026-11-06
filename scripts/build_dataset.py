import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.agronomy.crop_config import load_config, usable_value
from src.data_collection.weather_history import fetch_weather_range
from src.models.dataset_builder import build_dataset, validate_dataset, SOIL_PROPS, SOIL_DEPTHS

p = argparse.ArgumentParser()
p.add_argument("--districts", nargs="*")
p.add_argument("--first-year", type=int, default=2015)
p.add_argument("--last-year", type=int, default=2025)
p.add_argument("--out", default="data/processed/dataset_groundnut_surrogate.csv")
a = p.parse_args()

cal = pd.read_csv("data/reference/season_calendar_groundnut.csv")
cent = pd.read_csv("data/reference/district_centroids.csv")
if a.districts:
    cal = cal[cal["district"].isin(a.districts)]
tmax = usable_value(load_config(), "groundnut", "temperature_max_c")
if tmax is None:
    sys.exit("No verified T_Max in the config; cannot derive the heat label.")


def soil_fetcher(lat, lon):
    """Median of valid values on a 3x3 grid within +/-0.05 deg (~5 km) of the point.

    Skips masked (nodata) pixels such as built-up land. Assumption, not a TNAU rule.
    """
    import math
    import statistics
    from src.data_collection.soilgrids_client import fetch_soil_properties
    from src.preprocessing.soil_preprocessor import preprocess_record
    vals, n_ok = {}, 0
    for dy in (-0.05, 0.0, 0.05):
        for dx in (-0.05, 0.0, 0.05):
            try:
                res = fetch_soil_properties(lat + dy, lon + dx, SOIL_PROPS, SOIL_DEPTHS)
            except Exception:
                continue
            row = preprocess_record(res.to_dict())
            got = False
            for k, v in row.items():
                if (k.endswith("_mean") and not k.startswith("texture_")
                        and isinstance(v, float) and not math.isnan(v)):
                    vals.setdefault(k, []).append(v)
                    got = True
            n_ok += got
    out = {k: statistics.median(v) for k, v in vals.items()}
    out["soil_points_used"] = n_ok
    return out


df, skipped = build_dataset(cal, cent, range(a.first_year, a.last_year + 1), tmax,
                            fetch_weather_range, soil_fetcher)
print("rows:", len(df), "| skipped:", len(skipped))
for s in skipped[:15]:
    print("  skipped:", s)
print("validation problems:", validate_dataset(df) or "none")
if len(df):
    print(df["label_heat_flag"].value_counts(dropna=False).to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print("wrote", a.out)

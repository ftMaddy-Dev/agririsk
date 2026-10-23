import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.agronomy.crop_config import load_config
from src.agronomy.risk_rules import assess_all

p = argparse.ArgumentParser()
p.add_argument("--crop", required=True, choices=["sesame", "groundnut"])
p.add_argument("--stage")
p.add_argument("--lat", type=float, required=True)
p.add_argument("--lon", type=float, required=True)
p.add_argument("--csv", default="data/processed/soil_clean.csv")
p.add_argument("--max-temp", type=float)
p.add_argument("--standing-water-days", type=float)
a = p.parse_args()

df = pd.read_csv(a.csv)
row = df[(df.latitude.round(4) == round(a.lat, 4)) & (df.longitude.round(4) == round(a.lon, 4))]
if row.empty:
    sys.exit("No matching row in the processed CSV; fetch and preprocess that coordinate first.")
soil = {k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in row.iloc[0].to_dict().items()}
obs = {"max_temperature_c": a.max_temp, "standing_water_days": a.standing_water_days}
print(json.dumps(assess_all(load_config(), a.crop, a.stage, soil, obs), indent=2, default=str))

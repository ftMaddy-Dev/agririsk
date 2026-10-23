import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection.soilgrids_client import (
    fetch_soil_properties, save_json, save_csv, SoilGridsError)

p = argparse.ArgumentParser()
p.add_argument("--lat", type=float, required=True)
p.add_argument("--lon", type=float, required=True)
p.add_argument("--out-dir", default="data/raw")
a = p.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
try:
    r = fetch_soil_properties(a.lat, a.lon)
except SoilGridsError as e:
    sys.exit(f"ERROR: {e}")

tag = f"soilgrids_{a.lat:.4f}_{a.lon:.4f}"
print(save_json(r, Path(a.out_dir) / f"{tag}.json"))
print(save_csv(r, Path(a.out_dir) / f"{tag}.csv"))
for m in r.measurements:
    print(f"{m.property:6} {m.depth:9} {m.value} {m.unit} [{m.status}]")

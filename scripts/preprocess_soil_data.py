import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.soil_preprocessor import (
    preprocess_files, build_report, save_processed)

p = argparse.ArgumentParser()
p.add_argument("--input-dir", default="data/raw")
p.add_argument("--out", default="data/processed/soil_clean.csv")
a = p.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
paths = sorted(Path(a.input_dir).glob("soilgrids_*.json"))
if not paths:
    sys.exit(f"No soilgrids_*.json files in {a.input_dir}")
df = preprocess_files(paths)
report = build_report(df)
print(save_processed(df, report, a.out))
print(f"rows={report['rows']} value_columns={report['value_columns']}")
print("missing:", report["missing_by_column"] or "none")
print("out_of_range:", report["out_of_range_columns"] or "none")

"""Soil preprocessing: raw SoilGrids JSON -> clean wide table.

No agronomic interpretation here. Missing data is preserved, never zero-filled.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Physical plausibility limits (NOT agronomic thresholds). None = no upper limit.
PHYSICAL_BOUNDS = {
    "sand": (0, 100), "silt": (0, 100), "clay": (0, 100),
    "phh2o": (0, 14), "soc": (0, 1000), "nitrogen": (0, 1000),
    "cec": (0, None), "bdod": (0, None), "cfvo": (0, 100),
}
TEXTURE_SUM_TOLERANCE = 1.0  # assumption: rounding tolerance, percentage points

DECISIONS = [
    "Column names: <property>_<depth>_<statistic>, lowercase, e.g. sand_0_5cm_mean.",
    "Missing values kept as NaN (never zero); a <column>_status column keeps the reason.",
    "Values outside physical bounds set to NaN with status 'out_of_range'.",
    "No unit conversion: the retrieval client already converted stored values to standard units.",
    "Texture sum check (sand+silt+clay ~ 100) per depth, tolerance 1.0 point (assumed).",
    "All depths kept; selecting a depth for crop root zones awaits TNAU guidance.",
]


def norm_depth(depth: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", depth.lower()).strip("_")


def _in_bounds(prop: str, value: float) -> bool:
    lo, hi = PHYSICAL_BOUNDS.get(prop, (None, None))
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def preprocess_record(record: dict, source_file: str = "") -> dict:
    """Turn one raw SoilGrids result (dict) into one flat row."""
    row = {
        "latitude": record["latitude"],
        "longitude": record["longitude"],
        "retrieved_at_utc": record["retrieved_at_utc"],
        "source": record["source"],
        "source_file": source_file,
    }
    depths = set()
    for m in record["measurements"]:
        prop = m["property"].lower()
        depth = norm_depth(m["depth"])
        depths.add(depth)
        key = f"{prop}_{depth}_{m['statistic'].lower()}"
        value, status = m["value"], m["status"]
        if value is None or status != "ok":
            value = math.nan
        elif not _in_bounds(prop, value):
            logger.warning("%s=%s out of physical bounds; set to NaN", key, value)
            value, status = math.nan, "out_of_range"
        row[key] = value
        row[f"{key}_status"] = status

    for depth in sorted(depths):
        parts = [row.get(f"{p}_{depth}_mean") for p in ("sand", "silt", "clay")]
        if all(p is not None and not math.isnan(p) for p in parts):
            total = sum(parts)
            row[f"texture_sum_{depth}"] = total
            row[f"texture_sum_{depth}_ok"] = abs(total - 100) <= TEXTURE_SUM_TOLERANCE
        else:
            row[f"texture_sum_{depth}"] = math.nan
            row[f"texture_sum_{depth}_ok"] = None
    return row


def preprocess_files(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        record = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.append(preprocess_record(record, Path(p).name))
    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame) -> dict:
    value_cols = [c for c in df.columns
                  if c.endswith("_mean") and not c.startswith("texture_")]
    return {
        "rows": len(df),
        "value_columns": len(value_cols),
        "missing_by_column": {c: int(df[c].isna().sum()) for c in value_cols
                              if df[c].isna().any()},
        "out_of_range_columns": [c[:-7] for c in df.columns
                                 if c.endswith("_status")
                                 and (df[c] == "out_of_range").any()],
        "decisions": DECISIONS,
    }


def save_processed(df: pd.DataFrame, report: dict, out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_csv.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return out_csv

"""Schema and validation for observed yield records (label source, Module 5).

Labels derived from these records are 'observed_yield_derived', never 'observed risk'.
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ["crop", "district", "season", "year_start", "area_ha", "production_t",
            "yield_kg_ha", "source", "source_url", "retrieved_on"]
CROPS = {"sesame", "groundnut"}
YIELD_TOLERANCE = 0.05  # assumption: relative tolerance for rounding in published tables


def validate_yield_records(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; an empty list means the records passed."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    problems = []
    bad = sorted(set(df["crop"].str.lower()) - CROPS)
    if bad:
        problems.append(f"unknown crops: {bad}")
    for col in ("year_start", "area_ha", "production_t", "yield_kg_ha"):
        nums = pd.to_numeric(df[col], errors="coerce")
        if nums.isna().any():
            problems.append(f"{col}: {int(nums.isna().sum())} missing or non-numeric rows")
        if (nums < 0).any():
            problems.append(f"{col}: negative values")
    area = pd.to_numeric(df["area_ha"], errors="coerce")
    prod = pd.to_numeric(df["production_t"], errors="coerce")
    yld = pd.to_numeric(df["yield_kg_ha"], errors="coerce")
    ok = (area > 0) & prod.notna() & yld.notna()
    expected = prod * 1000 / area
    off = ok & ((yld - expected).abs() > YIELD_TOLERANCE * expected.clip(lower=1e-9))
    if off.any():
        problems.append(f"yield inconsistent with production/area in {int(off.sum())} rows "
                        "(check units: area in hectares, production in tonnes)")
    if ((area == 0) & (prod > 0)).any():
        problems.append("production > 0 with zero area")
    keys = ["crop", "district", "season", "year_start"]
    if df.duplicated(keys).any():
        problems.append(f"duplicate records for {keys}")
    if not df["source_url"].astype(str).str.startswith("https://").all():
        problems.append("source_url must be an https URL for every row")
    return problems


def temporal_split(df: pd.DataFrame, test_from_year: int):
    """Train on years before test_from_year, test on the rest (no random split)."""
    train = df[df["year_start"] < test_from_year]
    test = df[df["year_start"] >= test_from_year]
    if train.empty or test.empty:
        raise ValueError("temporal split produced an empty train or test set")
    return train, test

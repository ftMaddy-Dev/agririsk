import pandas as pd
import pytest
from src.models.dataset_schema import validate_yield_records, temporal_split


def rec(**kw):
    base = dict(crop="groundnut", district="Villupuram", season="kharif", year_start=2019,
                area_ha=1000.0, production_t=1500.0, yield_kg_ha=1500.0,
                source="test", source_url="https://example.org/x", retrieved_on="2026-11-06")
    base.update(kw)
    return base


def test_valid_records_pass():
    assert validate_yield_records(pd.DataFrame([rec()])) == []


def test_inconsistent_yield_flagged():
    p = validate_yield_records(pd.DataFrame([rec(yield_kg_ha=900.0)]))
    assert any("inconsistent" in s for s in p)


def test_missing_column_and_duplicates():
    assert "missing columns" in validate_yield_records(pd.DataFrame([{"crop": "sesame"}]))[0]
    df = pd.DataFrame([rec(), rec()])
    assert any("duplicate" in s for s in validate_yield_records(df))


def test_missing_values_not_zero_filled():
    p = validate_yield_records(pd.DataFrame([rec(production_t=None)]))
    assert any("production_t" in s for s in p)


def test_temporal_split_no_overlap():
    df = pd.DataFrame([rec(year_start=y) for y in (2018, 2019, 2020, 2021)])
    train, test = temporal_split(df, 2020)
    assert train.year_start.max() < test.year_start.min()
    with pytest.raises(ValueError):
        temporal_split(df, 2030)

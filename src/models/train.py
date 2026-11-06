"""Train and compare feature sets for the heat-flag surrogate label.

The label is a RULE-DERIVED SURROGATE, not an observed outcome: results measure
how well features reproduce the rule, not real heat damage.

Two evaluations:
  leave_one_year     : hold out one whole year (all districts together)
  leave_one_district : hold out one whole district (tests generalisation beyond
                       memorised districts)
Baselines: season_only (window start month) and district_only (district identity).
If the full models barely beat these, the extra features add nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.dataset_builder import LEAKY, SOIL_FEATURES, WEATHER_FEATURES

SEED = 42
REGIMES = ("irrigated", "rainfed", "unspecified")
SCHEMES = {"leave_one_year": "year_start", "leave_one_district": "district"}
CAVEAT = ("Label is rule-derived (surrogate), not an observed outcome; results show how well "
          "features reproduce the rule, not real heat damage. Positives are concentrated in "
          "a few district-seasons. Leave-one-year-out trains on later years too; not a "
          "forecasting test.")


def _slug(name: str) -> str:
    return re.sub(r"\W+", "_", str(name)).strip("_")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing label (once, for every configuration); add derived columns."""
    d = df[df["label_heat_flag"].notna()].copy()
    d["window_start_month"] = pd.to_datetime(d["window_start"]).dt.month
    for r in REGIMES:
        d[f"regime_{r}"] = (d["regime"] == r).astype(int)
    for dist in sorted(d["district"].unique()):
        d[f"dist_{_slug(dist)}"] = (d["district"] == dist).astype(int)
    return d.reset_index(drop=True)


def feature_sets(df: pd.DataFrame) -> dict:
    ctx_extra = ["window_start_month"] + [f"regime_{r}" for r in REGIMES]
    dist_cols = sorted(c for c in df.columns if c.startswith("dist_"))
    sets = {
        "season_only": ["window_start_month"],
        "district_only": dist_cols,
        "weather_only": list(WEATHER_FEATURES),
        "context_aware": list(WEATHER_FEATURES) + list(SOIL_FEATURES) + ctx_extra,
    }
    for name, cols in sets.items():
        assert not (set(cols) & LEAKY), f"{name} contains a label-leaking feature"
        sets[name] = [c for c in cols if c in df.columns and df[c].notna().any()]
    return sets


def make_models(pos_weight: float) -> dict:
    from xgboost import XGBClassifier
    return {
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=1),
        "xgboost": lambda: XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=SEED, n_jobs=1),
    }


def evaluate_oof(df: pd.DataFrame, cols: list, factory, group_col: str) -> np.ndarray:
    """Out-of-fold probability per row; each fold holds out one whole group."""
    X, y = df[cols], df["label_heat_flag"].astype(int)
    oof = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups=df[group_col]):
        ytr = y.iloc[tr]
        if ytr.nunique() < 2:                    # single-class training fold
            oof[te] = float(ytr.iloc[0])
            continue
        model = factory()
        model.fit(X.iloc[tr], ytr)
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def evaluate_loyo(df, cols, factory):
    return evaluate_oof(df, cols, factory, "year_start")


def evaluate_logo(df, cols, factory):
    return evaluate_oof(df, cols, factory, "district")


def metrics(y, p, threshold=0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    out = {"precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0)}
    two = len(set(y)) == 2
    out["roc_auc"] = roc_auc_score(y, p) if two else float("nan")
    out["pr_auc"] = average_precision_score(y, p) if two else float("nan")
    return {k: round(float(v), 3) for k, v in out.items()}


def run(dataset: str, out_dir: str, save: bool = True) -> dict:
    import joblib
    df = prepare(pd.read_csv(dataset))
    y = df["label_heat_flag"].astype(int)
    n_pos = int(y.sum())
    if n_pos < 5 or len(y) - n_pos < 5:
        raise SystemExit(f"Too few examples of one class ({n_pos} positive of {len(y)}); not training.")
    pos_weight = (len(y) - n_pos) / n_pos
    print(f"rows={len(df)} positives={n_pos} ({n_pos/len(df):.1%}) "
          f"years={df.year_start.nunique()} districts={df.district.nunique()}")
    print("positives per district:", df.groupby("district")["label_heat_flag"].sum().astype(int).to_dict())
    print("positives per year:", df.groupby("year_start")["label_heat_flag"].sum().astype(int).to_dict())

    sets, models = feature_sets(df), make_models(pos_weight)
    results = {"caveat": CAVEAT, "rows": len(df), "positives": n_pos, "runs": {}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for scheme, group_col in SCHEMES.items():
        print(f"\n=== {scheme} ===")
        for cfg, cols in sets.items():
            print(f"{cfg}: {len(cols)} features")
            for name, factory in models.items():
                oof = evaluate_oof(df, cols, factory, group_col)
                m = metrics(y, oof)
                results["runs"][f"{scheme}/{cfg}/{name}"] = {"features": cols, **m}
                print(f"  {name:14} " + "  ".join(f"{k}={v}" for k, v in m.items()))
    if save:
        for cfg, cols in sets.items():
            for name, factory in models.items():
                final = factory()
                final.fit(df[cols], y)
                joblib.dump({"model": final, "features": cols,
                             "label_origin": "rule_derived_surrogate"},
                            out / f"{cfg}__{name}.joblib")
        (out / "model_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nsaved models and model_comparison.json to", out)
    print("\nNOTE:", CAVEAT)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/dataset_groundnut_surrogate.csv")
    ap.add_argument("--out-dir", default="models_saved")
    a = ap.parse_args()
    run(a.dataset, a.out_dir)

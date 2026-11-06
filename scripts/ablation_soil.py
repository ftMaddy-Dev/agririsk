"""Context-aware with vs without soil, same folds and models. Surrogate label."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.models.dataset_builder import SOIL_FEATURES
from src.models.train import evaluate_oof, feature_sets, make_models, metrics, prepare

df = prepare(pd.read_csv("data/processed/dataset_groundnut_surrogate.csv"))
y = df["label_heat_flag"].astype(int)
full = feature_sets(df)["context_aware"]
no_soil = [c for c in full if c not in SOIL_FEATURES]
models = make_models((len(y) - y.sum()) / y.sum())
print("with soil   :", len(full), "features")
print("without soil:", len(no_soil), "features\n")
for scheme, grp in [("leave_one_year", "year_start"), ("leave_one_district", "district")]:
    print(f"=== {scheme} ===")
    for label, cols in [("with_soil", full), ("no_soil", no_soil)]:
        for name in ("random_forest", "xgboost"):
            m = metrics(y, evaluate_oof(df, cols, models[name], grp))
            print(f"  {label:10} {name:14} roc_auc={m['roc_auc']} pr_auc={m['pr_auc']} f1={m['f1']}")

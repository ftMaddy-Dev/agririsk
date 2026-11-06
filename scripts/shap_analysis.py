"""SHAP for the saved models. Explains what the FITTED model uses, not what causes heat damage.

Models were trained on all rows, so this is in-sample. The label is a rule-derived surrogate.
Soil is constant per district, so soil importance can mean 'district fingerprint'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.dataset_builder import SOIL_FEATURES, WEATHER_FEATURES
from src.models.train import prepare

MODELS = ["context_aware__xgboost", "context_aware__random_forest", "weather_only__xgboost"]
OUT = Path("models_saved")


def group_of(f):
    if f in SOIL_FEATURES:
        return "soil"
    if f in WEATHER_FEATURES:
        return "weather"
    return "season/regime"


def positive_class_shap(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):          # older shap: one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (rows, features, classes)
        sv = sv[:, :, 1]
    return sv


df = prepare(pd.read_csv("data/processed/dataset_groundnut_surrogate.csv"))
rows = []
for name in MODELS:
    path = OUT / f"{name}.joblib"
    if not path.exists():
        print("missing", path)
        continue
    bundle = joblib.load(path)
    cols = bundle["features"]
    X = df[cols]
    try:
        sv = positive_class_shap(bundle["model"], X)
    except Exception as exc:
        print(f"SKIPPED {name}: {type(exc).__name__}: {exc}")
        continue
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    print(f"\n=== {name}: mean |SHAP| (log-odds for XGBoost, probability for RF) ===")
    print(imp.round(4).to_string())
    grp = imp.groupby(imp.index.map(group_of)).sum()
    print("share by group:", (grp / grp.sum()).round(3).to_dict())
    for f, v in imp.items():
        rows.append({"model": name, "feature": f, "group": group_of(f), "mean_abs_shap": v})

    plt.figure(figsize=(7, 4))
    imp[::-1].plot.barh()
    plt.xlabel("mean |SHAP|")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(OUT / f"shap_bar_{name}.png", dpi=150)
    plt.close()

    if name == "context_aware__xgboost":
        shap.summary_plot(sv, X, show=False)
        plt.tight_layout()
        plt.savefig(OUT / f"shap_beeswarm_{name}.png", dpi=150)
        plt.close()

pd.DataFrame(rows).to_csv(OUT / "shap_summary.csv", index=False)
print("\nsaved plots and shap_summary.csv to", OUT)
print("NOTE: in-sample, surrogate label; soil importance may be a district fingerprint.")

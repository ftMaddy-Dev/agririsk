"""Is the soil gain bigger than noise? Leave-one-district, surrogate label.
1) 5 model seeds  2) district bootstrap of the PR-AUC difference (seed 42 models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.models.dataset_builder import SOIL_FEATURES
from src.models.train import evaluate_oof, feature_sets, make_models, metrics, prepare

df = prepare(pd.read_csv("data/processed/dataset_groundnut_surrogate.csv"))
y = df["label_heat_flag"].astype(int)
pw = (len(y) - y.sum()) / y.sum()
full = feature_sets(df)["context_aware"]
no_soil = [c for c in full if c not in SOIL_FEATURES]


def factory(name, seed):
    if name == "random_forest":
        return lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                              class_weight="balanced", random_state=seed, n_jobs=1)
    return lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 scale_pos_weight=pw, subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=seed, n_jobs=1)


print("=== 1) seeds: leave-one-district PR-AUC (with soil minus no soil) ===")
for name in ("random_forest", "xgboost"):
    diffs = []
    for seed in range(5):
        a = metrics(y, evaluate_oof(df, full, factory(name, seed), "district"))["pr_auc"]
        b = metrics(y, evaluate_oof(df, no_soil, factory(name, seed), "district"))["pr_auc"]
        diffs.append(a - b)
        print(f"  {name:14} seed={seed} with={a:.3f} without={b:.3f} diff={a-b:+.3f}")
    print(f"  {name:14} diff mean={np.mean(diffs):+.3f} sd={np.std(diffs):.3f} "
          f"min={min(diffs):+.3f} max={max(diffs):+.3f}\n")

print("=== 2) district bootstrap (1000 resamples, seed-42 models) ===")
models = make_models(pw)
groups = df["district"].to_numpy()
names = np.unique(groups)
idx_by = {d: np.where(groups == d)[0] for d in names}
rng = np.random.default_rng(0)
for name in ("random_forest", "xgboost"):
    pa = evaluate_oof(df, full, models[name], "district")
    pb = evaluate_oof(df, no_soil, models[name], "district")
    yy = y.to_numpy()
    d = []
    for _ in range(1000):
        pick = np.concatenate([idx_by[k] for k in rng.choice(names, len(names))])
        if yy[pick].sum() == 0:
            continue
        d.append(average_precision_score(yy[pick], pa[pick]) - average_precision_score(yy[pick], pb[pick]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {name:14} diff median={np.median(d):+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]  "
          f"share>0={np.mean(np.array(d) > 0):.2f}")

"""SHAP explanation for one prediction from a saved tree model.

Independent of FastAPI. Explains the actual trained model; contributions are
feature effects on the model output, not causal evidence.
"""
from __future__ import annotations

import numpy as np

from src.models.predict import feature_row, WARNING


def explain_one(bundle: dict, features: dict, top_k: int = 5) -> dict:
    import shap
    X, missing = feature_row(bundle, features)
    explainer = shap.TreeExplainer(bundle["model"])
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                       # older shap: one array per class
        vals = np.asarray(sv[1])[0]
    else:
        arr = np.asarray(sv)
        vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
    base = np.asarray(explainer.expected_value).ravel()
    base = float(base[1] if base.size > 1 else base[0])
    scale = "log-odds" if "xgb" in type(bundle["model"]).__name__.lower() else "probability"
    contrib = sorted(({"feature": c, "contribution": float(v),
                       "value": None if c in missing else float(X.iloc[0][c])}
                      for c, v in zip(X.columns, vals)),
                     key=lambda d: abs(d["contribution"]), reverse=True)
    return {"scale": scale, "base_value": base, "top_contributions": contrib[:top_k],
            "all_contributions": contrib, "missing_features": missing,
            "note": ("Contributions show how each feature moved this model's output; they are "
                     "not evidence of cause. " + WARNING)}

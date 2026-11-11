"""Load saved surrogate models and score one feature row.

The models are trained on a RULE-DERIVED label, not observed outcomes.
Missing features stay missing (NaN), never 0, and are reported back.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[2] / "models_saved"
THRESHOLD = 0.5   # assumption: default decision threshold, not tuned
WARNING = ("Surrogate model trained on a rule-derived label (daily maximum above the configured "
           "T_Max), not on observed outcomes. Preliminary decision support, not a validated "
           "prediction of crop damage.")


class ModelNotFoundError(Exception):
    pass


def list_models(model_dir: Path | str = MODEL_DIR) -> list[str]:
    return sorted(p.stem for p in Path(model_dir).glob("*.joblib"))


def load_bundle(name: str, model_dir: Path | str = MODEL_DIR) -> dict:
    """Only names that exist in the model directory are accepted (no path tricks)."""
    if name not in list_models(model_dir):
        raise ModelNotFoundError(f"Unknown model '{name}'. Available: {list_models(model_dir)}")
    return joblib.load(Path(model_dir) / f"{name}.joblib")


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def feature_row(bundle: dict, features: dict) -> tuple[pd.DataFrame, list[str]]:
    cols = bundle["features"]
    missing = [c for c in cols if _is_missing(features.get(c))]
    if len(missing) == len(cols):
        raise ValueError("None of the model's input features were provided")
    row = {c: (float("nan") if c in missing else float(features[c])) for c in cols}
    return pd.DataFrame([row], columns=cols), missing


def predict_one(bundle: dict, features: dict) -> dict:
    X, missing = feature_row(bundle, features)
    p = float(bundle["model"].predict_proba(X)[0, 1])
    return {"probability": round(p, 4),
            "predicted_flag": int(p >= THRESHOLD),
            "threshold": THRESHOLD,
            "label_origin": bundle.get("label_origin"),
            "features_used": {c: float(X.iloc[0][c]) for c in X.columns if c not in missing},
            "missing_features": missing,
            "warning": WARNING}

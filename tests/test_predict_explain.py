import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.models.predict import (ModelNotFoundError, list_models, load_bundle, predict_one,
                                WARNING)
from src.explainability.shap_explainer import explain_one

FEATS = ["f1", "f2"]


@pytest.fixture
def model_dir(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.random(80), "f2": rng.random(80)})
    y = (X["f1"] > 0.5).astype(int)
    m = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    joblib.dump({"model": m, "features": FEATS, "label_origin": "rule_derived_surrogate"},
                tmp_path / "toy__rf.joblib")
    return tmp_path


def test_unknown_or_path_name_rejected(model_dir):
    assert list_models(model_dir) == ["toy__rf"]
    for bad in ("../toy__rf", "nope", "/etc/passwd"):
        with pytest.raises(ModelNotFoundError):
            load_bundle(bad, model_dir)


def test_predict_reports_missing_not_zero(model_dir):
    b = load_bundle("toy__rf", model_dir)
    out = predict_one(b, {"f1": 0.9})
    assert out["missing_features"] == ["f2"] and "f2" not in out["features_used"]
    assert 0.0 <= out["probability"] <= 1.0 and out["warning"] == WARNING
    assert out["label_origin"] == "rule_derived_surrogate"
    with pytest.raises(ValueError):
        predict_one(b, {})


def test_explain_sorted_and_additive(model_dir):
    b = load_bundle("toy__rf", model_dir)
    feats = {"f1": 0.9, "f2": 0.2}
    e = explain_one(b, feats)
    mags = [abs(c["contribution"]) for c in e["all_contributions"]]
    assert mags == sorted(mags, reverse=True)
    total = e["base_value"] + sum(c["contribution"] for c in e["all_contributions"])
    assert total == pytest.approx(predict_one(b, feats)["probability"], abs=1e-3)
    assert "not evidence of cause" in e["note"]

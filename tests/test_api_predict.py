import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api.main import app
from src.models import predict as pm

client = TestClient(app)


@pytest.fixture(autouse=True)
def toy_models(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.random(100), "f2": rng.random(100)})
    y = (X["f1"] > 0.5).astype(int)
    m = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    joblib.dump({"model": m, "features": ["f1", "f2"], "label_origin": "rule_derived_surrogate"},
                tmp_path / "toy__rf.joblib")
    monkeypatch.setattr(pm, "MODEL_DIR", tmp_path)


def body(**kw):
    return {"crop": "groundnut", "model": "toy__rf", "features": {"f1": 0.95, "f2": 0.2}, **kw}


def test_models_lists():
    r = client.get("/models")
    assert r.status_code == 200 and r.json()["models"] == ["toy__rf"]


def test_predict_ok_and_missing_reported():
    r = client.post("/predict-risk", json=body(features={"f1": 0.95}))
    j = r.json()
    assert r.status_code == 200 and j["missing_features"] == ["f2"]
    assert j["basis"] == "surrogate_model" and "rule-derived" in j["warning"]


def test_unknown_model_404():
    assert client.post("/predict-risk", json=body(model="nope")).status_code == 404


def test_sesame_rejected():
    r = client.post("/predict-risk", json=body(crop="sesame"))
    assert r.status_code == 422 and "groundnut only" in r.json()["detail"]


def test_bad_features_422():
    assert client.post("/predict-risk", json=body(features={})).status_code == 422
    assert client.post("/predict-risk", json=body(features={"zzz": 1.0})).status_code == 422


def test_explain_returns_contributions():
    j = client.post("/explain-risk", json=body(top_k=2)).json()
    assert len(j["explanation"]["top_contributions"]) == 2
    assert "not evidence of cause" in j["explanation"]["note"]


def test_advisory_structure_and_sources():
    flagged = client.post("/generate-advisory", json=body(growth_stage="flowering")).json()
    assert flagged["risk_level"] == "flagged" and flagged["recommended_actions"]
    for a in flagged["recommended_actions"]:
        assert a["reference"].startswith("https://agritech.tnau.ac.in")
        assert "heat-stress-specific" in a["specificity"]
    assert "not evidence of cause" in flagged["contribution_note"]
    assert flagged["label_origin"] == "rule_derived_surrogate" and flagged["warning"]
    calm = client.post("/generate-advisory",
                       json=body(features={"f1": 0.05, "f2": 0.2})).json()
    assert calm["risk_level"] == "not_flagged" and calm["recommended_actions"] == []
    assert calm["no_action_reason"]

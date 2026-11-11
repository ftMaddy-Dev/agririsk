# AgriRisk

Preliminary decision support for sesame and groundnut risk, built for a final-year CSE project.
**All output is preliminary rule-based decision support. The machine learning label is a
rule-derived surrogate (a daily maximum above the configured T_Max), not an observed outcome.**
Model scores measure how well features reproduce that rule, not real crop damage.

## What is in it

| Part | Where | Status |
|---|---|---|
| Soil retrieval (SoilGrids 2.0, local Tamil Nadu window plus remote fallback) | `src/data_collection/soilgrids_client.py` | Working |
| Weather (Open-Meteo recent and archive) | `src/data_collection/weather_client.py`, `weather_history.py` | Working |
| Preprocessing and season features | `src/preprocessing/` | Working |
| Crop configuration with sourced thresholds (TNAU) | `data/reference/crop_config.json`, `src/agronomy/crop_config.py` | Partly resolved |
| Rule engine (five categories, returns "incomplete" when inputs or verified thresholds are missing) | `src/agronomy/risk_rules.py`, `POST /assess` | Working |
| Surrogate models (decision tree, random forest, XGBoost) | `src/models/`, `models_saved/` | Groundnut only |
| SHAP explanation | `src/explainability/shap_explainer.py` | Working |
| Advisory rules | `src/agronomy/advisory_rules.py` | General guidance only |
| Observation log | `src/observations/`, `/observations` | Needs API key |
| Frontend | not built | Deferred |

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest tests/ -q

The first install is large (xgboost pulls in about 200 MB of libraries). On a slow connection, rerun `pip install --resume-retries 10 --timeout 120 -r requirements.txt`.

Windows note: rasterio's bundled libraries were blocked by a Code Integrity policy on one machine;
development used WSL2 (Ubuntu).

## Run the API

Create `.env` with `AGRIRISK_API_KEY=<a long random string>` (it is git-ignored), then:

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --env-file .env

Docs: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
|---|---|
| `GET /soil` | SoilGrids properties for a coordinate |
| `POST /assess` | Rule-based assessment (five categories, may be incomplete) |
| `GET /models` | Saved surrogate models |
| `POST /predict-risk` | Surrogate model probability for a feature set (groundnut) |
| `POST /explain-risk` | SHAP contributions for that prediction |
| `POST /generate-advisory` | Advisory from prediction, explanation and guidance |
| `POST`/`GET /observations` | Field reports; require header `X-API-Key` |

## Data and scripts

`scripts/` holds the data build, training-related and analysis scripts (`build_dataset.py`,
`ablation_soil.py`, `seed_check.py`, `shap_analysis.py`, ...). Run any of them with `--help`.
Raw data, caches and the observation database are not in git. The saved surrogate models and SHAP outputs in `models_saved/` are committed, so the API and tests work from a clean clone.

## Limitations

- The label is a surrogate; no observed outcome data was available. Observed district yield was
  evaluated and not adopted (many values are state averages substituted for missing data).
- Soil features are 0-5 cm only, 250 m modelled estimates, aggregated around a town location.
- Weather is a modelled grid estimate, not a station reading.
- Open-Meteo's free tier is for non-commercial use and requires attribution ("Weather data by Open-Meteo.com").
- Soil, weather and models cover groundnut; sesame has configuration and rules only.
- Water stress and disease stay "incomplete": no verified numeric thresholds were found.
- API-key protection is one shared key over plain HTTP: acceptable locally, not for deployment.
- SoilGrids organic carbon scaling factor is unverified.

## Data sources

ISRIC SoilGrids 2.0 (CC BY 4.0); Open-Meteo; TNAU Crop Production Guide 2020 and Agritech portal.

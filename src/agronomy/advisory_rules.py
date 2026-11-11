"""Rule-based advisory text for a surrogate-model flag.

Only guidance traceable to the TNAU Crop Production Guide 2020 is used. Actions are
general guidance, not validated responses to heat stress; SHAP contributions are
never turned into causal claims.
"""
from __future__ import annotations

from src.models.predict import WARNING

CPG = "https://agritech.tnau.ac.in/pdf/AGRICULTURE.pdf"

GUIDANCE = {
    "groundnut": [{
        "id": "groundnut_critical_irrigation",
        "action": ("Do not miss irrigation at pegging, flowering and pod development; "
                   "adequate soil moisture is essential in these phases."),
        "reason": "The guide names these as the critical phases for irrigation of groundnut.",
        "reference": CPG + " (Crop Production Guide 2020, p.192, Water management)",
        "specificity": "General irrigation guidance; the guide gives no heat-stress-specific action.",
    }],
}

CONTRIBUTION_NOTE = ("Contributions show how each feature moved this model's output; they are "
                     "not evidence of cause.")


def build_advisory(crop: str, growth_stage: str | None, prediction: dict, explanation: dict) -> dict:
    flagged = prediction["predicted_flag"] == 1
    actions, no_action_reason = [], None
    if not flagged:
        no_action_reason = "The model did not flag this input; no advisory action is derived."
    elif crop not in GUIDANCE:
        no_action_reason = "No verified guidance is configured for this crop."
    else:
        actions = [dict(g) for g in GUIDANCE[crop]]
    return {
        "crop": crop,
        "growth_stage": growth_stage,
        "growth_stage_note": ("Recorded but not used: stage-wise thresholds are unresolved."
                              if growth_stage else None),
        "risk_category": "heat_flag_surrogate",
        "risk_level": "flagged" if flagged else "not_flagged",
        "probability": prediction["probability"],
        "threshold": prediction["threshold"],
        "label_rule": ("Flag = at least one day above the configured T_Max "
                       "(TNAU climate table, p.181); an assumption, not a damage threshold."),
        "contributing_features": explanation["top_contributions"],
        "contribution_note": CONTRIBUTION_NOTE,
        "contribution_scale": explanation.get("scale"),
        "contribution_base_value": explanation.get("base_value"),
        "recommended_actions": actions,
        "no_action_reason": no_action_reason,
        "uncertainty": {
            "missing_features": prediction["missing_features"],
            "evaluation": ("Single-run cross-validated scores are in models_saved/model_comparison.json; "
                           "they measure reproduction of the rule label, not real outcomes."),
        },
        "label_origin": prediction["label_origin"],
        "warning": WARNING,
    }

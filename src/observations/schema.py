"""Field observation report: the label source for any future model.

Stores what a person reported. No inference, no derived labels.
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Crop = Literal["sesame", "groundnut"]
Problem = Literal["none", "waterlogging", "heat_stress", "water_stress",
                  "disease", "nutrient_deficiency", "pest", "other"]
Severity = Literal["none", "mild", "moderate", "severe"]
Basis = Literal["reporter_observation", "extension_officer_confirmed",
                "lab_or_expert_confirmed"]
Role = Literal["farmer", "extension_officer", "researcher"]


class ObservationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop: Crop
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    observed_date: date
    growth_stage: str | None = Field(None, max_length=60)
    days_after_sowing: int | None = Field(None, ge=0, le=400)
    problem: Problem
    severity: Severity
    affected_area_pct: float | None = Field(None, ge=0, le=100)
    standing_water_days: float | None = Field(None, ge=0)
    diagnosis_basis: Basis = "reporter_observation"
    reporter_role: Role
    notes: str | None = Field(None, max_length=1000)

    @model_validator(mode="after")
    def _consistent(self):
        if self.observed_date > date.today():
            raise ValueError("observed_date cannot be in the future")
        if (self.problem == "none") != (self.severity == "none"):
            raise ValueError("problem 'none' and severity 'none' must go together")
        return self


class ObservationOut(ObservationIn):
    id: str
    created_at_utc: str

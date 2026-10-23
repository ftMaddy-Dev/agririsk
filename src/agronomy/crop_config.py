"""Load crop configuration. Thresholds live in data/reference/crop_config.json.

Only parameters with an accepted status are returned to rule logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "crop_config.json"
ACCEPTED = {"verified", "verified_excerpt"}
# Allowed source hosts per declared source_type. Extend only with real,
# checked publisher domains.
SOURCE_HOSTS = {
    "tnau": ("agritech.tnau.ac.in",),
    "icar": ("icar.gov.in", "icar.org.in"),
}


def load_config(path: str | Path = DEFAULT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_crop(config: dict, crop: str) -> dict:
    key = crop.strip().lower()
    if key not in config["crops"]:
        raise KeyError(f"Unknown crop '{crop}'. Available: {sorted(config['crops'])}")
    return config["crops"][key]


def get_parameter(config: dict, crop: str, name: str) -> dict:
    params = get_crop(config, crop)["parameters"]
    if name not in params:
        raise KeyError(f"Crop '{crop}' has no parameter '{name}'")
    return params[name]


def usable_value(config: dict, crop: str, name: str):
    """Value if verified, else None. Callers must treat None as 'unavailable'."""
    p = get_parameter(config, crop, name)
    return p["value"] if p["status"] in ACCEPTED and p["value"] is not None else None


def unresolved(config: dict, crop: str) -> list[str]:
    return [n for n, p in get_crop(config, crop)["parameters"].items()
            if p["status"] not in ACCEPTED]


def source_problem(param: dict) -> str | None:
    """None if the source URL matches the declared source_type, else a message."""
    st = param.get("source_type")
    if st not in SOURCE_HOSTS:
        return f"unknown or missing source_type {st!r}"
    host = urlparse(param.get("source", "")).hostname or ""
    if not any(host == h or host.endswith("." + h) for h in SOURCE_HOSTS[st]):
        return f"source host {host!r} does not match source_type {st!r}"
    return None

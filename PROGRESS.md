# Agririsk progress (23 Oct 2026)

Preliminary rule-based decision support. Not a validated model prediction.

## State
- Latest commit: df05fb2 (source_type with host check). main matches origin/main.
- 62 tests pass (1 known Starlette/httpx deprecation warning, not from our code).
- Git credentials: credential.helper store, ~/.git-credentials is chmod 600.

## Done
- Soil: local Tamil Nadu window, retry, cache, partial/nodata status.
- /assess: typed Pydantic response models, 422 on bad input.
- Weather: Open-Meteo client (src/data_collection/weather_client.py), completed
  days only, None never 0, retry on 5xx/429/network, no cache.
- Weather wired into /assess: weather block, include_weather flag (default true),
  heat_stress uses the highest daily max in the 7-day window with its date
  (a design choice, not a TNAU rule). Manual observation wins.
  Fetched inputs carry provenance "retrieved". Checked live: 31.9 C on 2026-10-17.
- Config: every sourced parameter has source_type (tnau or icar). The loader
  checks the source host against the declared type. No ICAR values added yet.

## Open (blocked on outside answers)
- TNAU email (draft written, send it): pH/texture/drainage limits, sesame
  waterlogging days, stage-wise limits, groundnut leaf spot model, whether
  40 C is a tolerance limit or a stress trigger, ET0 or rainfall water-stress
  threshold.
- water_stress and disease stay incomplete until verified thresholds exist.
- ICAR domains other than icar.gov.in and icar.org.in need adding when a real
  ICAR institute URL is rejected.

## Caveats
- Open-Meteo is a modelled grid estimate, not a station reading.
- ET0 is not pan evaporation, so IW/CPE ratios cannot be computed.
- Open-Meteo free tier is non-commercial only. Decide before any product use.
- Open-Meteo variable names and terms came from third-party summaries; verify at
  open-meteo.com/en/docs.
- data/raw/ is not in git. The soc factor is unverified.

## Next
1. Decide label source for the ML part (yield, damage reports, field
   observations, or none yet).
2. Dataset schema (Module 5), then labels, training, SHAP.
3. Later: MOSDAC rainfall, Sentinel imagery.

## Design rules
- No invented thresholds. Every value traces to a TNAU or ICAR source.
- Missing data stays None or NaN, never 0.
- Rule engine returns incomplete with the exact missing inputs.
- Waterlogging is never inferred from soil texture.
- All output is labelled preliminary rule-based decision support.
- Failed layers are absent from results; errors are never cached.

## How to work
- Terminal 1 runs only the server (no --reload; restart after code changes).
- Terminal 2 for everything else. Use cat heredocs or python patch scripts that
  end in py_compile, then check wc -l.

"""Season-window weather features from daily rows. Pure functions, no network.

Missing days are never treated as zero. Sums need a fully covered window;
means and maxima need MIN_COVERAGE.
"""
from __future__ import annotations

import datetime as dt
import math

MIN_COVERAGE = 0.9   # assumption: minimum share of valid days for means/maxima
NAN = float("nan")


def season_window(year_start: int, start_md: str, length_days: int):
    month, day = (int(x) for x in start_md.split("-"))
    start = dt.date(year_start, month, day)
    return start, start + dt.timedelta(days=length_days - 1)


def _vals(window, key):
    out = []
    for d in window:
        v = d.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(float(v))
    return out


def compute_season_features(days, start: dt.date, end: dt.date, hot_threshold_c=None) -> dict:
    window = [d for d in days if start <= dt.date.fromisoformat(d["date"]) <= end]
    n = (end - start).days + 1
    tmax, tmin = _vals(window, "tmax_c"), _vals(window, "tmin_c")
    rain, et0 = _vals(window, "precip_mm"), _vals(window, "et0_mm")
    cov = lambda v: len(v) / n
    ok = lambda v: cov(v) >= MIN_COVERAGE
    full = lambda v: len(v) == n

    f = {"window_start": start.isoformat(), "window_end": end.isoformat(), "window_days": n,
         "tmax_coverage": cov(tmax), "tmin_coverage": cov(tmin),
         "precip_coverage": cov(rain), "et0_coverage": cov(et0)}
    f["tmax_mean_c"] = sum(tmax) / len(tmax) if tmax and ok(tmax) else NAN
    f["tmax_max_c"] = max(tmax) if tmax and ok(tmax) else NAN
    f["tmin_mean_c"] = sum(tmin) / len(tmin) if tmin and ok(tmin) else NAN
    f["hot_days"] = (sum(1 for t in tmax if t > hot_threshold_c)
                     if hot_threshold_c is not None and full(tmax) else NAN)
    f["rain_sum_mm"] = sum(rain) if full(rain) else NAN
    f["rain_max_day_mm"] = max(rain) if rain and ok(rain) else NAN
    f["et0_sum_mm"] = sum(et0) if full(et0) else NAN
    f["rain_minus_et0_mm"] = (f["rain_sum_mm"] - f["et0_sum_mm"]
                              if full(rain) and full(et0) else NAN)
    return f

"""Show the crop state at a lat/lon: rule-engine assessment + nearby field reports.

Needs the API running (uvicorn src.api.main:app). Preliminary rule-based
decision support, not a validated model prediction.
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--crop", required=True, choices=["sesame", "groundnut"])
p.add_argument("--lat", type=float, required=True)
p.add_argument("--lon", type=float, required=True)
p.add_argument("--stage")
p.add_argument("--max-temp", type=float, help="manual max temperature, C")
p.add_argument("--standing-water-days", type=float)
p.add_argument("--radius-km", type=float, default=10.0)
p.add_argument("--no-weather", action="store_true")
p.add_argument("--base", default="http://127.0.0.1:8000")
a = p.parse_args()


def call(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"API error {e.code}: {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach API ({e.reason}). Is uvicorn running?")


def km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = (math.sin(dp / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


obs = {}
if a.max_temp is not None:
    obs["max_temperature_c"] = a.max_temp
if a.standing_water_days is not None:
    obs["standing_water_days"] = a.standing_water_days

body = {"crop": a.crop, "lat": a.lat, "lon": a.lon,
        "include_weather": not a.no_weather, "observations": obs}
if a.stage:
    body["growth_stage"] = a.stage
res = call("POST", f"{a.base}/assess", body)

print(f"\nCROP STATE: {a.crop} at ({a.lat}, {a.lon})")
print(res.get("disclaimer", "Preliminary rule-based decision support."))
for key in ("soil", "weather"):
    blk = res.get(key)
    if blk:
        print(f"{key}: status={blk.get('status')} {blk.get('message') or ''}".rstrip())

print(f"\n{'CATEGORY':18} {'STATUS':11} {'LEVEL':28} DETAIL")
for x in res.get("assessments", []):
    detail = "; ".join(x.get("reasons") or [])
    if x.get("status") == "incomplete" and x.get("missing_inputs"):
        detail = "missing: " + " | ".join(x["missing_inputs"])
    print(f"{x['category']:18} {x['status']:11} {str(x.get('level')):28} {detail[:110]}")
    for c in x.get("caveats") or []:
        if c:
            print(f"{'':18} caveat: {c[:110]}")

reports = call("GET", f"{a.base}/observations?crop={a.crop}&limit=1000")
near = sorted(
    ((km(a.lat, a.lon, r["lat"], r["lon"]), r) for r in reports),
    key=lambda t: t[0])
near = [(d, r) for d, r in near if d <= a.radius_km]
print(f"\nFIELD REPORTS within {a.radius_km:g} km: {len(near)}")
for d, r in near[:10]:
    print(f"  {r['observed_date']}  {d:5.1f} km  {r['problem']:12} "
          f"{r['severity']:9} [{r['diagnosis_basis']}]")
print()

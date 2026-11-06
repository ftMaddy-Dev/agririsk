"""Look up a main-town coordinate per district via Open-Meteo geocoding (GeoNames).

These are TOWN locations, not true district centroids. Review every line printed.
"""
import csv
import sys
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_collection.weather_client import _http_get_json

ALIASES = {"Thirunelveli": "Tirunelveli", "Thiruvannamalai": "Tiruvannamalai",
           "Kancheepuram": "Kanchipuram", "Thiruvallur": "Tiruvallur"}  # guesses; verify
cal = pd.read_csv("data/reference/season_calendar_groundnut.csv")
out = []
for d in sorted(cal["district"].unique()):
    q = urlencode({"name": ALIASES.get(d, d), "count": 10, "country_code": "IN"})
    res = _http_get_json(f"https://geocoding-api.open-meteo.com/v1/search?{q}").get("results") or []
    hit = next((r for r in res if r.get("admin1") == "Tamil Nadu"), None)
    if hit is None:
        print(f"NOT FOUND: {d}")
        continue
    print(f"{d:16} -> {hit['name']}, {hit['admin1']}  {hit['latitude']:.4f} {hit['longitude']:.4f}")
    out.append([d, hit["latitude"], hit["longitude"],
                f"Open-Meteo geocoding (GeoNames) town '{hit['name']}'; not a district centroid"])
with open("data/reference/district_centroids.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["district", "lat", "lon", "source"])
    w.writerows(out)
print(len(out), "districts written")

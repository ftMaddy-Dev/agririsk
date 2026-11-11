import re
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/yield_tamilnadu.csv"
d = pd.read_csv(path)
print("columns:", list(d.columns))
print("rows:", len(d), "| fiscal_year:", d["fiscal_year"].min(), "to", d["fiscal_year"].max())
print("seasons:", sorted(d["season"].dropna().unique()))

crops = sorted(d["crop"].dropna().unique())
hits = [c for c in crops if re.search(r"groundnut|sesam|gingelly|til|oilseed", str(c), re.I)]
print("\noilseed-like crop names:", hits)

ours = ["Coimbatore", "Cuddalore", "Dindigul", "Erode", "Kancheepuram", "Karur", "Madurai",
        "Pudukkottai", "Ramanathapuram", "Sivagangai", "Theni", "Thirunelveli", "Thiruvallur",
        "Thiruvannamalai", "Thoothukudi", "Tiruppur", "Vellore", "Villupuram", "Virudhunagar"]
alias = {"thirunelveli": "tirunelveli", "thiruvallur": "tiruvallur",
         "thiruvannamalai": "tiruvannamalai", "sivagangai": "sivaganga"}


def norm(s):
    s = re.sub(r"[^a-z]", "", str(s).lower())
    return alias.get(s, s)


col = "district_as_per_lgd" if "district_as_per_lgd" in d.columns else "district_as_per_source"
d["_k"] = d[col].map(norm)
want = {norm(x): x for x in ours}
print("\nour districts NOT found in the file:",
      [v for k, v in want.items() if k not in set(d["_k"])])

for crop in [c for c in hits if re.search(r"groundnut|sesam|gingelly", str(c), re.I)]:
    s = d[(d["crop"] == crop) & d["_k"].isin(want)].copy()
    s["yr"] = s["fiscal_year"].astype(str).str[:4].astype(int)
    s = s[(s["yr"] >= 2015) & (s["yr"] <= 2025)]
    print(f"\n=== {crop}: rows by season ===")
    print(s.groupby("season")["crop_yield"].agg(["count", lambda x: x.notna().sum()]).rename(
        columns={"<lambda_0>": "non_null"}).to_string())
    m = s.pivot_table(index=s["_k"].map(want), columns="yr", values="crop_yield", aggfunc="count")
    print("district x year (count of season rows):")
    print(m.fillna(0).astype(int).to_string())

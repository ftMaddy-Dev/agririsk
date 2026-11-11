import re
from pathlib import Path

import pandas as pd

NUM8 = r"(\d+(?:\s+\d+){7})"
COLS = ["c69", "c70", "g_a", "g_b", "g_c", "s_a", "s_b", "s_c"]


def parse_text(text: str, year: str) -> pd.DataFrame:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    inline = []
    for l in lines:
        m = re.match(rf"^(?:\d+\.?\s+)?([A-Za-z][A-Za-z .]*?)\s+{NUM8}$", l)
        if m:
            inline.append((m.group(1).strip(), [int(x) for x in m.group(2).split()]))
    if inline:
        rows = inline
    else:
        names = [re.sub(r"^\d+\.?\s*", "", l) for l in lines
                 if re.match(r"^\d+\.?\s+[A-Za-z]", l)] + ["State"] * any(l == "State" for l in lines)
        nums = [[int(x) for x in l.split()] for l in lines
                if re.fullmatch(NUM8, l) and l.split()[:2] != ["69", "70"]]
        if len(names) != len(nums):
            print(f"  WARNING {year}: {len(names)} names but {len(nums)} number rows. "
                  f"Aligning the first {min(len(names), len(nums))}.")
        rows = list(zip(names, nums))
    df = pd.DataFrame([[n] + v for n, v in rows], columns=["district"] + COLS)
    df.insert(0, "year", year)
    return df


if __name__ == "__main__":
    d = Path("data/raw/season_crop_text")
    frames = []
    for f in sorted(d.glob("*_p*.txt")):
        year = f.name.split("_p")[0]
        t = f.read_text(encoding="utf-8")
        if "AVERAGE YIELD RATE" not in t.upper():
            continue
        df = parse_text(t, year)
        print(f"{f.name}: {len(df)} rows")
        frames.append(df)
    out = pd.concat(frames)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/processed/yield_table_raw.csv", index=False)
    st = out[out.district == "State"].set_index("year")
    for y, g in out[out.district != "State"].groupby("year"):
        if y not in st.index:
            print(f"{y}: no State row parsed")
            continue
        for c in ["g_a", "g_b", "g_c", "s_a", "s_b", "s_c"]:
            n = (g[c] == st.loc[y, c]).sum()
            print(f"{y} {c}: equals the state value in {n} of {len(g)} districts")

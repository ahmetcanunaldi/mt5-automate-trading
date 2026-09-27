"""EXP-066: trading-day-of-month calendar sweep on D1 2008-2026 (open->close, valid on D1): day number from the
start (1..8) and from the end (-1..-6) of the month; 19 years, era consistency, Bonferroni-aware reading."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet").loc["2008":]
r = np.log(d.close / d.open) * 1e4
mon = d.index.to_period("M")
fwd = pd.Series(1, index=d.index).groupby(mon).cumcount() + 1
bwd = -(pd.Series(1, index=d.index).groupby(mon).cumcount(ascending=False) + 1)
ERAS = [(2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026)]
rows = []
for lab_, s in (("from start", fwd), ("from end", bwd)):
    for k in (list(range(1, 9)) if lab_ == "from start" else list(range(-6, 0))):
        v = r[s == k]
        er = [round(v[(v.index.year >= a) & (v.index.year <= b)].mean(), 1) for a, b in ERAS]
        rows.append({"day": f"{lab_} {k:+d}", "n": len(v), "bps": round(v.mean(), 1),
                     "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2), "eras_pos": sum(e > 0 for e in er), "eras": er})
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.to_string(index=False))
print("baseline all days:", round(r.mean(), 1), "bps; 14 tests -> Bonferroni |t| > 2.9")

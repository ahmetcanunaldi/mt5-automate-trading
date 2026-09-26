"""EXP-048: (a) weekend gap / Monday behaviour, (b) pre-holiday drift, (c) gold/silver ratio mean reversion (daily)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
r_oc = np.log(d.close / d.open) * 1e4
gap = np.log(d.open / d.close.shift(1)) * 1e4


def stat(name, v):
    v = v.dropna()
    yr = v.groupby(v.index.year).mean()
    print(f"{name:<52} n {len(v):>5} mean {v.mean():+7.2f} bps  t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}  yrs+ {(yr > 0).sum()}/{len(yr)}")


print("=== (a) weekend gap & Monday ===")
mon = d.index.dayofweek == 0
stat("Monday gap (Fri close -> Mon open)", gap[mon])
stat("Monday open->close", r_oc[mon])
stat("Monday open->close signed by -gap (fade gap)", (-np.sign(gap) * r_oc)[mon])
stat("Monday open->close signed by gap (follow gap)", (np.sign(gap) * r_oc)[mon & (gap.abs() > 20)])
stat("Mon fade when |gap| > 30 bps", (-np.sign(gap) * r_oc)[mon & (gap.abs() > 30)])

print("\n=== (b) pre-/post-holiday (days before a >=3 day market gap) ===")
nxt_gap = pd.Series(d.index, index=d.index).shift(-1) - pd.Series(d.index, index=d.index)
pre_hol = (nxt_gap > pd.Timedelta(days=1)) & (d.index.dayofweek != 4)
stat("pre-holiday (non-Friday) open->close", r_oc[pre_hol])
post = pd.Series(pre_hol.shift(1, fill_value=False).to_numpy(), index=d.index)
stat("post-holiday open->close", r_oc[post])

print("\n=== (c) gold/silver ratio (2018+) ===")
xs = pd.read_parquet(lab.DATA / "XAGUSD_M1_2018.parquet")["close"].resample("1D").last().dropna()
g = d.close.loc["2018-10":]
ratio = np.log(g / xs.reindex(g.index)).dropna()
z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
fwd5 = np.log(d.close.shift(-5) / d.open.shift(-1)).reindex(z.index) * 1e4
for thr in (1.5, 2.0):
    stat(f"gold 5d fwd when ratio z < -{thr} (gold cheap)", fwd5[z < -thr])
    stat(f"gold 5d fwd when ratio z > +{thr} (gold rich)", fwd5[z > thr])
stat("gold 5d fwd unconditional", fwd5)

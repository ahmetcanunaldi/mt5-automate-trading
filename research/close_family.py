"""EXP-058: close-location family on D1 2008-2026 (next-day open->close, valid on D1) and on M1 2019-2026.
  weak close short (all / below EMA50), strong close long split by trend, 2-3 day holds, weekly strong close ->
  next Monday / next week long."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.features import atr, ema  # noqa: E402

d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
a = atr(d, 14)
clv = ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low).replace(0, np.nan)
e50 = ema(d.close, 50); up = d.close > e50
o1 = d.open.shift(-1)
fwd = {h: np.log(d.close.shift(-h) / o1) * 1e4 for h in (1, 2, 3, 5)}
ERAS = [(2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026)]


def stat(name, cond, h=1, sign=1):
    v = (sign * fwd[h][cond.fillna(False)]).dropna()
    er = [round(v[(v.index.year >= x) & (v.index.year <= y)].mean(), 1) for x, y in ERAS]
    print(f"{name:<52} h{h} n {len(v):>4} {v.mean():+6.1f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f} eras {er}")


print("=== daily close location (D1 2008-2026, next-day open -> close after h days) ===")
for q in (0.6, 0.8):
    stat(f"strong close > {q} (long)", clv > q)
    stat(f"strong close > {q} & uptrend (long)", (clv > q) & up)
    stat(f"strong close > {q} & downtrend (long)", (clv > q) & ~up)
    stat(f"weak close < -{q} (short)", clv < -q, sign=-1)
    stat(f"weak close < -{q} & downtrend (short)", (clv < -q) & ~up, sign=-1)
    for h in (2, 3, 5):
        stat(f"strong close > {q} (long)", clv > q, h=h)
print("\n=== weekly close location -> next week ===")
w = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
wclv = ((w.close - w.low) - (w.high - w.close)) / (w.high - w.low)
mon = d[d.index.dayofweek == 0]
mon_ret = np.log(mon.close / mon.open) * 1e4
nxt_week = np.log(w.close.shift(-1) / w.open.shift(-1)) * 1e4
wk_prev = wclv.reindex(mon.index - pd.Timedelta(days=3))
for q in (0.6, 0.8):
    v = mon_ret[(wk_prev > q).to_numpy()]
    print(f"weekly clv > {q} -> Monday long: n {len(v)} {v.mean():+.1f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}")
    v = nxt_week[wclv > q].dropna()
    er = [round(v[(v.index.year >= x) & (v.index.year <= y)].mean(), 1) for x, y in ERAS]
    print(f"weekly clv > {q} -> next week long: n {len(v)} {v.mean():+.1f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f} eras {er}")
    v = -nxt_week[wclv < -q].dropna()
    print(f"weekly clv < -{q} -> next week short: n {len(v)} {v.mean():+.1f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}")

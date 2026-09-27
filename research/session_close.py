"""EXP-059: intraday session close-location -> next session (M1 2019-2026, server time).
Sessions: Asia 01:05-10:00, London 10:00-16:30, NY-AM 16:30-20:00, NY-PM 20:00-23:30.
Condition on the session's close location in its own range (clv) and on the session's return; measure the NEXT
session's return (bps), per-year consistency; both 'follow strength' and 'fade' directions."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

S = {"asia": (65, 600), "london": (600, 990), "ny_am": (990, 1200), "ny_pm": (1200, 1410)}
m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2019":]
tm = m1.index.hour * 60 + m1.index.minute
day = m1.index.normalize()
stats = {}
for s, (a, b) in S.items():
    x = m1[(tm >= a) & (tm < b)]
    g = x.groupby(x.index.normalize())
    stats[s] = pd.DataFrame({"o": g.open.first(), "h": g.high.max(), "l": g.low.min(), "c": g.close.last()})
names = list(S)
for i in range(len(names) - 1):
    cur, nxt = stats[names[i]], stats[names[i + 1]]
    j = cur.join(nxt, rsuffix="_n", how="inner")
    clv = ((j.c - j.l) - (j.h - j.c)) / (j.h - j.l).replace(0, np.nan)
    r_n = np.log(j.c_n / j.o_n) * 1e4
    for q in (0.6, 0.8):
        for name, cond, sign in ((f"{names[i]} clv>{q} -> {names[i+1]} long", clv > q, 1),
                                 (f"{names[i]} clv<-{q} -> {names[i+1]} short", clv < -q, -1),
                                 (f"{names[i]} clv<-{q} -> {names[i+1]} long (fade)", clv < -q, 1)):
            v = (sign * r_n[cond]).dropna()
            yr = v.groupby(v.index.year).mean()
            print(f"{name:<44} n {len(v):>4} {v.mean():+5.2f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f} "
                  f"yrs+ {(yr > 0).sum()}/{len(yr)}")

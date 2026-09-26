"""EXP-046: regime check of the breakout legs on daily bars 2008-2026 (covers the 2013-18 bear/range).
Daily-bar simulation (pessimistic): stop-entry at the trigger level; if the same day also touches the stop,
the trade is a full loss; otherwise exit at the close. Cost $0.50/oz round trip (older, wider spreads).
  lw     : levels open +/- k * prev_range, SL = sl * ATR_D from the trigger
  inside : after an inside day, break of the prior high/low, SL = other side (cap 1.5 ATR), trend filter
  nr7    : same after an NR7 day
  tday*  : not testable on D1 (needs intraday path)
Also Friday-long and turn-of-month as calendar controls."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.features import atr, ema  # noqa: E402

COST = 0.50


def run(d, kind, k=0.4, sl=1.0, trend=True):
    a = atr(d, 14).shift(1)
    rng = (d.high - d.low)
    prev_rng = rng.shift(1); pdh = d.high.shift(1); pdl = d.low.shift(1)
    e20, e50 = ema(d.close, 20), ema(d.close, 50)
    st = pd.Series(np.where((d.close > e50) & (e20 > e50), 1, np.where((d.close < e50) & (e20 < e50), -1, 0)),
                   index=d.index).shift(1)
    nr7 = (rng == rng.rolling(7).min()).shift(1)
    inside = ((d.high < d.high.shift(1)) & (d.low > d.low.shift(1))).shift(1)
    out = []
    for i in range(260, len(d)):
        dt = d.index[i]; o, h, l, c = d.open.iat[i], d.high.iat[i], d.low.iat[i], d.close.iat[i]
        A = a.iat[i]
        if not (A > 0):
            continue
        if kind == "lw":
            up, dn = o + k * prev_rng.iat[i], o - k * prev_rng.iat[i]
        elif kind in ("inside", "nr7"):
            if not bool((inside if kind == "inside" else nr7).iat[i]):
                continue
            up, dn = pdh.iat[i], pdl.iat[i]
        hit_up, hit_dn = h >= up, l <= dn
        if not (hit_up or hit_dn):
            continue
        if hit_up and hit_dn:
            out.append((dt, -1.0 - COST / (sl * A)))      # order unknown -> pessimistic full loss
            continue
        dirn = 1 if hit_up else -1
        if trend and st.iat[i] != dirn:
            continue
        entry = max(up, o) if dirn == 1 else min(dn, o)
        risk = sl * A if kind == "lw" else min(abs(entry - (pdl.iat[i] if dirn == 1 else pdh.iat[i])), 1.5 * A)
        if risk <= 0:
            continue
        stop = entry - dirn * risk
        stopped = (l <= stop) if dirn == 1 else (h >= stop)
        r = -1.0 if stopped else (c - entry) * dirn / risk
        out.append((dt, r - COST / risk))
    return pd.Series(dict(out))


def weekday_tom(d):
    a = atr(d, 14).shift(1)
    r_day = (d.close - d.open - COST) / (1.5 * a)
    fri = r_day[d.index.dayofweek == 4].clip(lower=-1)
    return fri


if __name__ == "__main__":
    d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
    tests = {"lw k0.4 sl1": run(d, "lw", 0.4, 1.0, False), "lw k0.4 sl1 trend": run(d, "lw", 0.4, 1.0, True),
             "lw k0.6 sl1": run(d, "lw", 0.6, 1.0, False), "inside trend": run(d, "inside", trend=True),
             "nr7 trend": run(d, "nr7", trend=True), "inside no-trend": run(d, "inside", trend=False),
             "friday long": weekday_tom(d)}
    rows = []
    for name, r in tests.items():
        r = r.dropna()
        eras = {f"{a}-{b}": r[(r.index.year >= a) & (r.index.year <= b)] for a, b in
                ((2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026))}
        rows.append({"test": name, "n": len(r), "avgR": round(r.mean(), 3), "t": round(r.mean() / r.std() * np.sqrt(len(r)), 2),
                     **{f"R {k}": round(v.mean(), 3) for k, v in eras.items()},
                     **{f"t {k}": round(v.mean() / v.std() * np.sqrt(len(v)), 2) for k, v in eras.items()}})
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    (lab.REPORTS / "EXP-046").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-046" / "summary.csv", index=False)

"""EXP-081: daily-bar pattern battery on the long D1 histories (EURUSD 1990-, USDJPY 2000-, XAGUSD 2012-, XAUUSD
2007- as reference). Next-day open->close (valid on D1) or next-5-day close-to-close for momentum states; signed
so that the tested direction is positive; 4 equal eras; cost in bps of one round trip at the median price.
Tests per symbol: weekday (L/S), month (L/S), first/last trading days of month, close location (top/bottom decile,
follow and fade), streaks 3/4 (follow/fade), big-range days (follow/fade), 20-day range position (follow/fade),
EMA20/50 trend state next day and next 5 days, TSMOM 60/250 next 5 days."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab, symbols  # noqa: E402
from research.features import atr, ema  # noqa: E402

COST_BPS = {"XAUUSD": 1.5, "XAGUSD": 14.6, "EURUSD": 2.2, "USDJPY": 2.2}


def battery(sym):
    d = pd.read_parquet(lab.DATA / f"{sym}_D1_long.parquet")[["open", "high", "low", "close"]]
    d = d[d.index.dayofweek < 5]
    d = d[(d.high > d.low)]
    a = atr(d, 20)
    nxt = (np.log(d.close / d.open) * 1e4).shift(-1)
    nxt5 = (np.log(d.close.shift(-5) / d.close) * 1e4)
    ret = np.log(d.close / d.close.shift(1)) * 1e4
    rng = (d.high - d.low) / a.shift(1)
    clv = ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)
    up = (ret > 0).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    streak = streak.where(up == 1, -streak)
    pos20 = (d.close - d.low.rolling(20).min()) / (d.high.rolling(20).max() - d.low.rolling(20).min())
    e20, e50 = ema(d.close, 20), ema(d.close, 50)
    state = pd.Series(np.where((d.close > e50) & (e20 > e50), 1, np.where((d.close < e50) & (e20 < e50), -1, 0)), index=d.index)
    y0, y1 = d.index.year.min() + 1, d.index.year.max()
    edges = np.linspace(y0, y1 + 1, 5).astype(int)
    eras = list(zip(edges[:-1], edges[1:] - 1))
    rows = []

    def stat(name, cond, sign=1, fwd=nxt):
        v = (sign * fwd[cond.fillna(False).astype(bool)]).loc[str(y0):].dropna()
        if len(v) < 60:
            return
        er = [v[(v.index.year >= a_) & (v.index.year <= b_)].mean() for a_, b_ in eras]
        rows.append({"sym": sym, "test": name, "n": len(v), "bps": round(v.mean(), 1),
                     "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2), "eras_pos": int(sum(e > 0 for e in er)),
                     "eras": [round(e, 1) for e in er], "net_bps": round(v.mean() - COST_BPS[sym], 1)})

    nd = pd.Series(np.roll(d.index.dayofweek, -1), index=d.index)
    for dw in range(5):
        stat(f"weekday {dw} long", nd == dw); stat(f"weekday {dw} short", nd == dw, -1)
    nm = pd.Series(d.index.month, index=d.index).shift(-1)
    for mo in range(1, 13):
        stat(f"month {mo:02d} long", nm == mo); stat(f"month {mo:02d} short", nm == mo, -1)
    per = d.index.to_period("M")
    fpos = pd.Series(1, index=d.index).groupby(per).cumcount().shift(-1)
    bpos = pd.Series(1, index=d.index).groupby(per).cumcount(ascending=False).shift(-1)
    for k in (0, 1):
        stat(f"month day +{k + 1} long", fpos == k); stat(f"month day +{k + 1} short", fpos == k, -1)
    stat("last day of month long", bpos == 0); stat("last day of month short", bpos == 0, -1)
    for q in (0.8,):
        stat("strong close -> long", clv > q); stat("strong close -> short (fade)", clv > q, -1)
        stat("weak close -> short", clv < -q, -1); stat("weak close -> long (fade)", clv < -q)
    for s in (3, 4):
        stat(f"{s}+ up days follow", streak >= s); stat(f"{s}+ up days fade", streak >= s, -1)
        stat(f"{s}+ down days follow", streak <= -s, -1); stat(f"{s}+ down days fade", streak <= -s)
    stat("range>2ATR up day follow", (rng > 2) & (ret > 0)); stat("range>2ATR up day fade", (rng > 2) & (ret > 0), -1)
    stat("range>2ATR down day follow", (rng > 2) & (ret < 0), -1); stat("range>2ATR down day fade", (rng > 2) & (ret < 0))
    stat("20d high zone follow", pos20 > 0.9); stat("20d high zone fade", pos20 > 0.9, -1)
    stat("20d low zone follow", pos20 < 0.1, -1); stat("20d low zone fade", pos20 < 0.1)
    stat("trend state next day", state != 0, fwd=nxt * state)
    stat("trend state next 5d", state != 0, fwd=nxt5 * state)
    for L in (60, 250):
        m = np.sign(np.log(d.close / d.close.shift(L)))
        stat(f"tsmom{L} next 5d", m != 0, fwd=nxt5 * m)
    return rows


if __name__ == "__main__":
    rows = []
    for sym in ("XAUUSD", "XAGUSD", "EURUSD", "USDJPY"):
        rows += battery(sym)
    T = pd.DataFrame(rows)
    (lab.REPORTS / "EXP-081").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-081" / "daily_multi.csv", index=False)
    pd.set_option("display.width", 220)
    ntests = T.groupby("sym").size().to_dict()
    print("tests per symbol:", ntests, "-> Bonferroni |t| > 3.4 (~70 tests)")
    for s, g in T.groupby("sym"):
        good = g[(g.t > 2.0) & (g.eras_pos >= 3)].sort_values("t", ascending=False)
        print(f"\n=== {s} (t > 2 and >= 3/4 eras positive): {len(good)}")
        print(good.head(15).to_string(index=False))

"""Swing (multi-day) signal library on daily bars. Signals are indexed by decision time = the day's close
(index + 1 day), so entry happens at the next daily open. Exits: SL, TP, trailing (chandelier on daily
extremes), max hold. "state" strategies emit a signal every day the condition holds; the engine only
enters when flat, so a position closed by a stop/weekend is re-entered while the state persists.
"""
import numpy as np
import pandas as pd

from research.features import atr, ema


def _sig(df, long_, short, sl, tp, hold_days, trail=0.0, leg=""):
    a = df["atr"]
    rows = []
    for m, d in ((long_, 1), (short, -1)):
        m = m.fillna(False).to_numpy(bool)
        if not m.any():
            continue
        rows.append(pd.DataFrame({"dir": d, "sl": (sl * a).to_numpy()[m], "tp": (tp * a).to_numpy()[m],
                                  "hold_min": int(hold_days * df.attrs.get("bar", pd.Timedelta(days=1)).total_seconds() / 60), "be": 0.0,
                                  "trail": (trail * a).to_numpy()[m] if trail else 0.0, "leg": leg},
                                 index=df.index[m] + df.attrs.get("bar", pd.Timedelta(days=1))))
    if not rows:
        return pd.DataFrame(columns=["dir", "sl", "tp", "hold_min", "be", "trail", "leg"])
    s = pd.concat(rows).sort_index(kind="stable")
    s = s[(s.sl > 0) & np.isfinite(s.sl)]
    return s[~s.index.duplicated(keep="first")]


def prep(d, bar=None):
    df = d.copy()
    if bar is not None:
        df.attrs["bar"] = bar
    df["atr"] = atr(df, 20)
    for n in (10, 20, 50, 100, 200):
        df[f"ema{n}"] = ema(df.close, n)
    df["ret20"] = df.close.pct_change(20)
    df["ret60"] = df.close.pct_change(60)
    df["ret120"] = df.close.pct_change(120)
    delta = df.close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    df["rsi2"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ma20 = df.close.rolling(20).mean(); sd20 = df.close.rolling(20).std()
    df["bb_lo"] = ma20 - 2 * sd20; df["bb_hi"] = ma20 + 2 * sd20; df["ma20"] = ma20
    wk = df.resample("W-FRI").agg({"high": "max", "low": "min"}).shift(1)
    df["pwh"] = wk["high"].reindex(df.index, method="ffill"); df["pwl"] = wk["low"].reindex(df.index, method="ffill")
    return df


# ---------------- trend following ----------------
def donchian(df, n=20, sl=2.0, trail=3.0, tp=20.0, hold=40, filt=None, both=True):
    hi = df.high.rolling(n).max().shift(1); lo = df.low.rolling(n).min().shift(1)
    L = df.close > hi; S = df.close < lo
    if filt == "ema200":
        L &= df.close > df.ema200; S &= df.close < df.ema200
    if not both:
        S = S & False
    return _sig(df, L, S, sl, tp, hold, trail, f"donchian{n}")


def ema_state(df, fast=20, slow=100, sl=2.0, trail=3.0, tp=20.0, hold=20, both=True):
    L = df[f"ema{fast}"] > df[f"ema{slow}"]
    S = (df[f"ema{fast}"] < df[f"ema{slow}"]) & both
    return _sig(df, L, S, sl, tp, hold, trail, f"ema{fast}_{slow}")


def tsmom(df, look="ret60", sl=2.0, trail=0.0, tp=20.0, hold=5, both=True, weekly=True):
    L = df[look] > 0; S = (df[look] < 0) & both
    if weekly:                                  # decide on Fridays, hold the week
        fri = pd.Series(df.index.dayofweek == 4, index=df.index)
        L &= fri; S &= fri
    return _sig(df, L, S, sl, tp, hold, trail, f"tsmom_{look}")


def week_break(df, sl=1.5, trail=2.5, tp=20.0, hold=10, filt="ema100"):
    L = (df.close > df.pwh) & (df.close.shift(1) <= df.pwh)
    S = (df.close < df.pwl) & (df.close.shift(1) >= df.pwl)
    if filt:
        L &= df.close > df[filt]; S &= df.close < df[filt]
    return _sig(df, L, S, sl, tp, hold, trail, "week_break")


# ---------------- mean reversion ----------------
def rsi2(df, lo=10, hi=90, sl=2.5, tp=1.5, hold=5, filt="ema200", both=True):
    L = (df.rsi2 < lo) & (df.close > df[filt])
    S = (df.rsi2 > hi) & (df.close < df[filt]) & both
    return _sig(df, L, S, sl, tp, hold, 0.0, "rsi2")


def bollinger_mr(df, sl=2.5, tp=1.5, hold=5, filt="ema200", both=True):
    L = (df.close < df.bb_lo) & (df.close > df[filt])
    S = (df.close > df.bb_hi) & (df.close < df[filt]) & both
    return _sig(df, L, S, sl, tp, hold, 0.0, "bb_mr")


# ---------------- calendar ----------------
def turn_of_month(df, days_before=1, days_after=3, sl=2.5):
    """Long from the close `days_before`+1 trading days before month end, hold through `days_after` days."""
    idx = df.index
    month = idx.to_period("M")
    pos_from_end = pd.Series(1, index=idx).groupby(month).cumcount(ascending=False)  # 0 = last day of month
    L = pd.Series(pos_from_end.to_numpy() == days_before, index=idx)
    return _sig(df, L, L & False, sl, 50.0, days_before + days_after + 1, 0.0, "tom")


def weekday(df, dow=0, direction=1, sl=2.0):
    """Hold one day: enter at the open of weekday `dow` (decision = previous day close)."""
    nxt = pd.Series(np.roll(df.index.dayofweek, -1), index=df.index)
    m = nxt == dow
    return _sig(df, m & (direction == 1), m & (direction == -1), sl, 50.0, 1, 0.0, f"dow{dow}")


LIB = {
    "donchian": donchian, "ema_state": ema_state, "tsmom": tsmom, "week_break": week_break,
    "rsi2": rsi2, "bollinger_mr": bollinger_mr, "turn_of_month": turn_of_month, "weekday": weekday,
}

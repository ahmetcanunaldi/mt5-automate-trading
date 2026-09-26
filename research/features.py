"""Feature helpers. All features at row t use information up to the CLOSE of bar t only.

Server-time landmarks (NY-close server, UTC+2/+3 follows US DST):
  01:00 daily open after break, Asia 01:00-10:00, London open ~10:00, US data 15:30, NY cash open 16:30.
"""
import numpy as np
import pandas as pd


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def htf_to_ltf(ltf_index, htf: pd.DataFrame, htf_minutes: int, cols):
    """Map higher-timeframe values known at HTF bar close onto lower-timeframe bars (by LTF close time)."""
    h = htf[cols].copy()
    h.index = h.index + pd.Timedelta(minutes=htf_minutes)  # available at close
    t = pd.DatetimeIndex(np.asarray(ltf_index))
    left = pd.DataFrame({"_t": t})
    out = pd.merge_asof(left, h.sort_index(), left_on="_t", right_index=True, direction="backward")
    return out.drop(columns="_t").set_index(t)


def base_frame(m15: pd.DataFrame, h1: pd.DataFrame, bar_minutes=15) -> pd.DataFrame:
    df = m15[["open", "high", "low", "close", "tick_volume", "spread"]].copy()
    df["close_time"] = df.index + pd.Timedelta(minutes=bar_minutes)
    df["atr"] = atr(df, 14)
    df["atr_slow"] = atr(df, 96)
    df["date"] = df.index.normalize()
    df["tmin"] = df.index.hour * 60 + df.index.minute
    df["dow"] = df.index.dayofweek
    # H1 context
    h = h1.copy()
    h["h1_ema50"] = ema(h["close"], 50)
    h["h1_ema200"] = ema(h["close"], 200)
    h["h1_atr"] = atr(h, 14)
    h["h1_close"] = h["close"]
    ctx = htf_to_ltf(df["close_time"], h, 60, ["h1_ema50", "h1_ema200", "h1_atr", "h1_close"])
    ctx.index = df.index
    df = df.join(ctx)
    # daily levels (previous server day)
    d = df.groupby("date").agg(dh=("high", "max"), dl=("low", "min"), dc=("close", "last"))
    d = d.shift(1)
    df = df.join(d, on="date")
    # session VWAP (from 01:00 server) using tick volume
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["tick_volume"].clip(lower=1)
    g = df["date"]
    df["vwap"] = (tp * v).groupby(g).cumsum() / v.groupby(g).cumsum()
    # Asia range 01:00-10:00 known at 10:00
    asia = df[(df["tmin"] >= 60) & (df["tmin"] < 600)]
    ar = asia.groupby("date").agg(asia_hi=("high", "max"), asia_lo=("low", "min"))
    df = df.join(ar, on="date")
    df.loc[df["tmin"] < 600, ["asia_hi", "asia_lo"]] = np.nan
    return df


def to_signals(df, mask_long, mask_short, sl, tp, hold_min, be=0.0, trail=0.0):
    """Build a signals frame indexed by decision time (bar close). sl/tp/be: scalar or Series on df.index."""
    def pick(v, m):
        if np.isscalar(v):
            return np.full(m.sum(), float(v))
        return pd.Series(v, index=df.index).to_numpy(float)[m]

    rows = []
    for mask, d in ((mask_long, 1), (mask_short, -1)):
        m = mask.fillna(False).to_numpy(bool)
        if not m.any():
            continue
        rows.append(pd.DataFrame({"dir": d, "sl": pick(sl, m), "tp": pick(tp, m),
                                  "hold_min": hold_min, "be": pick(be, m), "trail": pick(trail, m)},
                                 index=pd.DatetimeIndex(df["close_time"].to_numpy()[m])))
    if not rows:
        return pd.DataFrame(columns=["dir", "sl", "tp", "hold_min", "be", "trail"], index=pd.DatetimeIndex([]))
    s = pd.concat(rows).sort_index()
    s = s[(s["sl"] > 0) & (s["tp"] > 0) & np.isfinite(s["sl"]) & np.isfinite(s["tp"])]
    return s[~s.index.duplicated(keep="first")]

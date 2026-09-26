"""Rule-based intraday signal families (M15 decision bars, H1 context).

Every family enforces the user's minimum target: TP >= MIN_TP_USD ($6).
Functions return a signals frame (see features.to_signals).
"""
import numpy as np
import pandas as pd

from research.features import ema, to_signals

MIN_TP_USD = 6.0


def _first_per_day(df, mask):
    m = mask.fillna(False)
    first = m & (m.groupby(df["date"]).cumsum() == 1)
    return first


def _trend(df, mode):
    if mode == "none":
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    up = (df["h1_ema50"] > df["h1_ema200"]) & (df["h1_close"] > df["h1_ema50"])
    dn = (df["h1_ema50"] < df["h1_ema200"]) & (df["h1_close"] < df["h1_ema50"])
    if mode == "with":
        return up, dn
    if mode == "against":
        return dn, up
    raise ValueError(mode)


def asia_breakout(df, start=600, end=780, sl_atr=1.5, rr=1.5, min_rng_atr=1.0, max_rng_atr=6.0,
                  trend="none", hold_min=240, be_r=0.0, sl_mode="atr"):
    """A: London-open breakout of the Asia (01:00-10:00 server) range."""
    rng = df["asia_hi"] - df["asia_lo"]
    ok_rng = (rng > min_rng_atr * df["atr"]) & (rng < max_rng_atr * df["atr"])
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    up, dn = _trend(df, trend)
    long_ = _first_per_day(df, win & ok_rng & (df["close"] > df["asia_hi"]) & up)
    short = _first_per_day(df, win & ok_rng & (df["close"] < df["asia_lo"]) & dn)
    if sl_mode == "atr":
        sl = sl_atr * df["atr"]
    else:  # beyond range mid
        sl = (rng / 2).clip(lower=df["atr"])
    tp = np.maximum(MIN_TP_USD, rr * sl)
    return to_signals(df, long_, short, sl, tp, hold_min, be_r * sl if be_r else 0.0)


def ny_orb(df, or_start=990, or_bars=2, end=1200, sl_atr=1.5, rr=1.5, trend="none", hold_min=180,
           min_rng_atr=0.5, max_rng_atr=5.0):
    """B: New York cash-open (16:30 server) opening-range breakout."""
    or_end = or_start + 15 * or_bars
    in_or = (df["tmin"] >= or_start) & (df["tmin"] < or_end)
    orh = df["high"].where(in_or).groupby(df["date"]).transform("max")
    orl = df["low"].where(in_or).groupby(df["date"]).transform("min")
    rng = orh - orl
    ok = (rng > min_rng_atr * df["atr"]) & (rng < max_rng_atr * df["atr"])
    win = (df["tmin"] >= or_end) & (df["tmin"] < end)
    up, dn = _trend(df, trend)
    long_ = _first_per_day(df, win & ok & (df["close"] > orh) & up)
    short = _first_per_day(df, win & ok & (df["close"] < orl) & dn)
    sl = sl_atr * df["atr"]
    tp = np.maximum(MIN_TP_USD, rr * sl)
    return to_signals(df, long_, short, sl, tp, hold_min)


def ema_pullback(df, fast=20, sl_atr=1.5, rr=2.0, start=600, end=1260, hold_min=240, touch_atr=0.3,
                 max_per_day=2):
    """C: H1 trend + M15 pullback to EMA(fast) and bullish/bearish rejection close."""
    e = ema(df["close"], fast)
    up, dn = _trend(df, "with")
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    body_up = df["close"] > df["open"]
    body_dn = df["close"] < df["open"]
    long_ = win & up & (df["low"] <= e + touch_atr * df["atr"]) & (df["close"] > e) & body_up
    short = win & dn & (df["high"] >= e - touch_atr * df["atr"]) & (df["close"] < e) & body_dn
    long_ = long_ & (long_.groupby(df["date"]).cumsum() <= max_per_day)
    short = short & (short.groupby(df["date"]).cumsum() <= max_per_day)
    sl = sl_atr * df["atr"]
    tp = np.maximum(MIN_TP_USD, rr * sl)
    return to_signals(df, long_, short, sl, tp, hold_min)


def vwap_reversion(df, dev_atr=2.5, sl_atr=1.0, tp_frac=0.8, start=180, end=1200, hold_min=180,
                   max_trend_slope=None):
    """D: fade large deviations from session VWAP; TP a fraction of the way back (only if >= $6)."""
    dev = df["close"] - df["vwap"]
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    tp_dist = tp_frac * dev.abs()
    ok = win & (tp_dist >= MIN_TP_USD)
    if max_trend_slope is not None:
        slope = (df["h1_ema50"] - df["h1_ema50"].shift(16)).abs() / df["h1_atr"]
        ok &= slope < max_trend_slope
    short = ok & (dev > dev_atr * df["atr"]) & (df["close"] < df["open"])
    long_ = ok & (dev < -dev_atr * df["atr"]) & (df["close"] > df["open"])
    long_ = _first_per_day(df, long_)
    short = _first_per_day(df, short)
    sl = sl_atr * df["atr"] + (df["high"] - df["low"])
    return to_signals(df, long_, short, sl, tp_dist, hold_min)


def prev_day_break(df, start=600, end=1200, sl_atr=1.5, rr=1.5, trend="none", hold_min=240, buffer_atr=0.1):
    """E: breakout of the previous server-day high/low."""
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    up, dn = _trend(df, trend)
    long_ = _first_per_day(df, win & up & (df["close"] > df["dh"] + buffer_atr * df["atr"])
                           & (df["open"] <= df["dh"] + buffer_atr * df["atr"]))
    short = _first_per_day(df, win & dn & (df["close"] < df["dl"] - buffer_atr * df["atr"])
                           & (df["open"] >= df["dl"] - buffer_atr * df["atr"]))
    sl = sl_atr * df["atr"]
    tp = np.maximum(MIN_TP_USD, rr * sl)
    return to_signals(df, long_, short, sl, tp, hold_min)


FAMILIES = {
    "asia_breakout": asia_breakout,
    "ny_orb": ny_orb,
    "ema_pullback": ema_pullback,
    "vwap_reversion": vwap_reversion,
    "prev_day_break": prev_day_break,
}

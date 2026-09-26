"""Rule-based intraday signal families (M15 decision bars, H1 context).

Each family returns entry masks + a stop distance; `build` applies a common exit scheme:
  tp = max(MIN_TP_USD, rr * sl)           (user rule: every trade targets >= $6)
  trail = trail_atr * ATR (0 = off), be = be_r * sl (0 = off), hold_min = max holding time.
"""
import numpy as np
import pandas as pd

from research.features import ema, to_signals

MIN_TP_USD = 6.0


def _first_per_day(df, mask, k=1):
    m = mask.fillna(False)
    return m & (m.groupby(df["date"]).cumsum() <= k) & m


def _trend(df, mode):
    t = pd.Series(True, index=df.index)
    if mode == "none":
        return t, t
    up = (df["h1_ema50"] > df["h1_ema200"]) & (df["h1_close"] > df["h1_ema50"])
    dn = (df["h1_ema50"] < df["h1_ema200"]) & (df["h1_close"] < df["h1_ema50"])
    if mode == "with":
        return up, dn
    if mode == "against":
        return dn, up
    if mode == "long_only":
        return t, ~t
    raise ValueError(mode)


def build(df, long_, short, sl, rr=1.5, trail_atr=0.0, be_r=0.0, hold_min=240, min_sl_usd=2.0, max_sl_usd=40.0):
    sl = pd.Series(sl, index=df.index).clip(lower=min_sl_usd)
    ok = sl <= max_sl_usd
    tp = np.maximum(MIN_TP_USD, rr * sl)
    trail = trail_atr * df["atr"] if trail_atr else 0.0
    be = be_r * sl if be_r else 0.0
    return to_signals(df, long_ & ok, short & ok, sl, tp, hold_min, be, trail)


def asia_breakout(df, start=600, end=780, sl_atr=1.5, min_rng_atr=1.0, max_rng_atr=6.0, trend="none", **ex):
    """A: London-open breakout of the Asia (01:00-10:00 server) range."""
    rng = df["asia_hi"] - df["asia_lo"]
    ok = (rng > min_rng_atr * df["atr"]) & (rng < max_rng_atr * df["atr"])
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    up, dn = _trend(df, trend)
    long_ = _first_per_day(df, win & ok & (df["close"] > df["asia_hi"]) & up)
    short = _first_per_day(df, win & ok & (df["close"] < df["asia_lo"]) & dn)
    return build(df, long_, short, sl_atr * df["atr"], **ex)


def ny_orb(df, or_start=990, or_bars=2, end=1200, sl_atr=1.5, trend="none", min_rng_atr=0.5, max_rng_atr=5.0, **ex):
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
    return build(df, long_, short, sl_atr * df["atr"], **ex)


def ema_pullback(df, fast=20, sl_atr=1.5, start=600, end=1260, touch_atr=0.3, per_day=2, trend="with", **ex):
    """C: H1 trend + M15 pullback to EMA(fast) and rejection close."""
    e = ema(df["close"], fast)
    up, dn = _trend(df, trend)
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    long_ = win & up & (df["low"] <= e + touch_atr * df["atr"]) & (df["close"] > e) & (df["close"] > df["open"])
    short = win & dn & (df["high"] >= e - touch_atr * df["atr"]) & (df["close"] < e) & (df["close"] < df["open"])
    return build(df, _first_per_day(df, long_, per_day), _first_per_day(df, short, per_day), sl_atr * df["atr"], **ex)


def donchian(df, n=16, sl_atr=1.5, start=600, end=1200, trend="with", per_day=2, min_width_atr=2.0, **ex):
    """F: M15 Donchian(n) channel breakout with H1 trend filter."""
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    ok = (hi - lo) > min_width_atr * df["atr"]
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    up, dn = _trend(df, trend)
    long_ = win & ok & up & (df["close"] > hi) & (df["open"] <= hi)
    short = win & ok & dn & (df["close"] < lo) & (df["open"] >= lo)
    return build(df, _first_per_day(df, long_, per_day), _first_per_day(df, short, per_day), sl_atr * df["atr"], **ex)


def vwap_reversion(df, dev_atr=2.5, sl_atr=1.0, start=180, end=1200, max_slope=None, **ex):
    """D: fade large deviations from session VWAP (TP at >= $6 by construction via rr)."""
    dev = df["close"] - df["vwap"]
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    ok = win
    if max_slope is not None:
        slope = (df["h1_ema50"] - df["h1_ema50"].shift(16)).abs() / df["h1_atr"]
        ok = ok & (slope < max_slope)
    short = _first_per_day(df, ok & (dev > dev_atr * df["atr"]) & (df["close"] < df["open"]))
    long_ = _first_per_day(df, ok & (dev < -dev_atr * df["atr"]) & (df["close"] > df["open"]))
    return build(df, long_, short, sl_atr * df["atr"] + (df["high"] - df["low"]), **ex)


def prev_day_break(df, start=600, end=1200, sl_atr=1.5, trend="none", buffer_atr=0.1, **ex):
    """E: breakout of the previous server-day high/low."""
    win = (df["tmin"] >= start) & (df["tmin"] < end)
    up, dn = _trend(df, trend)
    lvl_h = df["dh"] + buffer_atr * df["atr"]
    lvl_l = df["dl"] - buffer_atr * df["atr"]
    long_ = _first_per_day(df, win & up & (df["close"] > lvl_h) & (df["open"] <= lvl_h))
    short = _first_per_day(df, win & dn & (df["close"] < lvl_l) & (df["open"] >= lvl_l))
    return build(df, long_, short, sl_atr * df["atr"], **ex)


FAMILIES = {
    "asia_breakout": asia_breakout,
    "ny_orb": ny_orb,
    "ema_pullback": ema_pullback,
    "donchian": donchian,
    "vwap_reversion": vwap_reversion,
    "prev_day_break": prev_day_break,
}

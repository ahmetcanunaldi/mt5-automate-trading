"""Data layer of the quant-math branch: resampled bars, log returns, aligned multi-symbol panels, realized measures.

Lockbox: 2025-01-01 .. 2026-09-25 is reserved for the final, one-shot evaluation of candidates. Every loader
cuts at DEV_END unless `lockbox=True` is passed explicitly (and that call should appear only in phase-7 scripts).
"""
import numpy as np
import pandas as pd

from research import symbols
from research.fresh_era import AGG

SYMS = ["XAUUSD", "XAGUSD", "NAS100", "DJ30", "SP500", "GER40", "EURUSD", "USDJPY"]
DEV_START, DEV_END = "2018-01-01", "2024-12-31"
LOCK_START, LOCK_END = "2025-01-01", "2026-09-25"
_CACHE: dict = {}


def m1(sym, lockbox=False):
    key = (sym, lockbox)
    if key not in _CACHE:
        d = symbols.load_m1(sym)
        d = d[d.index.dayofweek < 5]
        _CACHE[key] = d.loc[DEV_START:(LOCK_END if lockbox else DEV_END)]
    return _CACHE[key]


def bars(sym, tf="1h", lockbox=False):
    """OHLCV bars (tick_volume summed, spread averaged) resampled from M1; empty bins dropped."""
    if tf in ("1min", "1m"):
        return m1(sym, lockbox)
    b = m1(sym, lockbox).resample(tf).agg(AGG).dropna(subset=["open"])
    return b[b.index.dayofweek < 5]


def log_returns(sym, tf="1h", lockbox=False, intraday_only=False):
    """Close-to-close log returns. intraday_only drops returns that span a day change (overnight / weekend gaps)."""
    b = bars(sym, tf, lockbox)
    r = np.log(b.close).diff()
    if intraday_only:
        r = r[b.index.normalize() == pd.Series(b.index, index=b.index).shift(1).dt.normalize()]
    return r.dropna()


def panel(syms, tf="1h", lockbox=False):
    """Aligned close-to-close log returns on common timestamps (inner join)."""
    return pd.concat({s: np.log(bars(s, tf, lockbox).close) for s in syms}, axis=1, join="inner").diff().dropna()


def realized(sym, tf="5min", lockbox=False):
    """Daily realized variance (sum of squared intraday tf returns), bipower variation (jump-robust),
    jump share, daily open->close return and tick volume. Server days, weekdays only."""
    b = bars(sym, tf, lockbox)
    r = np.log(b.close).diff()
    same = b.index.normalize() == pd.Series(b.index, index=b.index).shift(1).dt.normalize()
    r = r[same]
    day = r.index.normalize()
    rv = (r ** 2).groupby(day).sum()
    ar = r.abs()
    bv = (np.pi / 2) * (ar * ar.groupby(day).shift(1)).groupby(day).sum()
    d = b.groupby(b.index.normalize()).agg(open=("open", "first"), close=("close", "last"), vol=("tick_volume", "sum"))
    out = pd.DataFrame({"rv": rv, "bv": bv}).join(d, how="inner")
    out["jump"] = (out.rv - out.bv).clip(lower=0) / out.rv.replace(0, np.nan)
    out["ret"] = np.log(out.close / out.open)
    return out

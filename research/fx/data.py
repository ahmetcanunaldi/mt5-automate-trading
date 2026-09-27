"""FX majors panel for the breadth ("Renaissance-style") edge search.

Universe: 8 currencies (USD, EUR, GBP, JPY, CHF, AUD, CAD, NZD) traded through the 7 USD majors. Any currency-level
weight vector w (per currency, vs USD) is realised exactly with the majors: XXXUSD position = +w_XXX,
USDXXX position = -w_XXX. NZDUSD history on the broker starts 2026-01-02; before that it is synthesised as
AUDUSD / AUDNZD on common H1 timestamps.
Periods: DEV = data start .. 2024-12-31, HOLDOUT = 2025-01-01 .. 2026-09-25 (one-shot, final candidates only)."""
import numpy as np
import pandas as pd

from research import lab

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD"]          # NZDUSD: broker history starts 2026 (AUDNZD too) -> excluded
CCY = {"EURUSD": "EUR", "GBPUSD": "GBP", "AUDUSD": "AUD", "NZDUSD": "NZD", "USDJPY": "JPY", "USDCHF": "CHF", "USDCAD": "CAD"}
SIGN = {p: (1.0 if p.endswith("USD") else -1.0) for p in PAIRS}           # currency-vs-USD = SIGN * pair
CCYS = ["USD"] + [CCY[p] for p in PAIRS]
POINT = {p: (0.001 if p == "USDJPY" else 0.00001) for p in PAIRS}
DEV_END, HOLD_START, HOLD_END = "2024-12-31", "2025-01-01", "2026-09-25"
COMMISSION = 5.0             # USD per lot round turn
SLIP_PTS = 3.0               # per side
_C: dict = {}


def raw(sym, tf="H1"):
    k = (sym, tf)
    if k not in _C:
        d = pd.read_parquet(lab.DATA / f"{sym}_{tf}_hist.parquet")
        _C[k] = d[d.index.dayofweek < 5]
    return _C[k]


def pair(sym, tf="H1"):
    if sym != "NZDUSD":
        return raw(sym, tf)
    real = raw("NZDUSD", tf)
    au, an = raw("AUDUSD", tf), raw("AUDNZD", tf)
    ix = au.index.intersection(an.index)
    ix = ix[ix < real.index[0]]
    syn = pd.DataFrame({"open": au.open[ix] / an.open[ix], "close": au.close[ix] / an.close[ix],
                        "high": au.high[ix] / an.close[ix], "low": au.low[ix] / an.close[ix],
                        "tick_volume": au.tick_volume[ix], "spread": np.nan}, index=ix)
    syn["high"] = syn[["high", "open", "close"]].max(1); syn["low"] = syn[["low", "open", "close"]].min(1)
    return pd.concat([syn, real])


def closes(tf="H1", start=None, end=HOLD_END):
    """pair closes on the union of timestamps, forward-filled (gaps of a pair are rare)."""
    P = pd.concat({p: pair(p, tf).close for p in PAIRS}, axis=1).sort_index().ffill().dropna()
    return P.loc[start:end]


def ccy_log(tf="H1", **kw):
    """log value of each currency in USD (USD = 0)."""
    P = np.log(closes(tf, **kw))
    L = pd.DataFrame({CCY[p]: SIGN[p] * P[p] for p in PAIRS})
    L.insert(0, "USD", 0.0)
    return L


def spread_bp(tf="H1"):
    """median recorded spread in bp of price per pair and calendar year (NZDUSD pre-2026: 2026 median)."""
    out = {}
    for p in PAIRS:
        d = pair(p, tf)
        bp = d.spread * POINT[p] / d.close * 1e4
        s = bp.groupby(d.index.year).median()
        out[p] = s.reindex(range(d.index.year.min(), d.index.year.max() + 1)).bfill().ffill()
    return pd.DataFrame(out)


def cost_bp(p, year, spreads=None):
    """round-turn cost in bp: recorded spread + 2 x slippage + commission."""
    spreads = spread_bp() if spreads is None else spreads
    px = pair(p, "D1").close.median()
    notional = 100_000.0 * (px if p.endswith("USD") else 1.0)
    return float(spreads.loc[year, p] + 2 * SLIP_PTS * POINT[p] / px * 1e4 + COMMISSION / notional * 1e4)

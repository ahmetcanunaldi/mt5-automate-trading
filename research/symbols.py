"""Symbol specifications (Vantage, read via MetaTrader5.symbol_info on 2026-09-27) and per-symbol cost presets.

Swaps are USD per lot per night (swap_mode = points: points x tick value; USDJPY tick value ~0.636 USD at 157).
Commission $7/lot round turn for every symbol (conservative). Slippage per side in points.
News: USD high-impact for all four symbols, plus EUR (ECB, EU CPI, DE GDP) for EURUSD and JPY (BoJ, JP data) for USDJPY.
"""
import numpy as np
import pandas as pd

from research import calendar_news, engine, lab

SPECS = {
    "XAUUSD": dict(point=0.01, contract=100.0, quote="USD", news=("usd",)),
    "XAGUSD": dict(point=0.001, contract=5000.0, quote="USD", news=("usd",)),
    "EURUSD": dict(point=0.00001, contract=100_000.0, quote="USD", news=("usd", "eur")),
    "USDJPY": dict(point=0.001, contract=100_000.0, quote="JPY", news=("usd", "jpy")),
}
COSTS = {
    "XAUUSD": engine.Costs(),
    "XAGUSD": engine.Costs(min_spread_pts=15.0, slippage_pts=5.0, swap_long=-119.15, swap_short=10.70),
    "EURUSD": engine.Costs(min_spread_pts=5.0, slippage_pts=3.0, swap_long=-5.76, swap_short=2.50),
    "USDJPY": engine.Costs(min_spread_pts=5.0, slippage_pts=3.0, swap_long=4.30, swap_short=-13.68),
}
CAL = {"usd": "calendar_usd_high.csv", "eur": "calendar_eur_high.csv", "jpy": "calendar_jpy_high.csv"}


def load_m1(sym):
    return pd.read_parquet(lab.DATA / f"{sym}_M1_2018.parquet")


def news_times(sym):
    parts = [calendar_news.load_news_server_times(lab.DATA / CAL[k]) for k in SPECS[sym]["news"]]
    return pd.DatetimeIndex(np.unique(np.concatenate([p.values for p in parts])))


def prepare(sym, bars, before=30, after=30, flatten=10, costs=None):
    news = news_times(sym)
    blk, flt = calendar_news.blackout_masks(bars.index, 1, news, before, after, flatten)
    sp = SPECS[sym]
    return engine.prepare_exec(bars, 1, blk, flt, costs or COSTS[sym], point=sp["point"], contract=sp["contract"],
                               quote=sp["quote"])

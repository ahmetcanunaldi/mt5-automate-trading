"""Symbol specifications (Vantage, read via MetaTrader5.symbol_info on 2026-09-27) and per-symbol cost presets.

Swaps are USD per lot per night (swap_mode = points: points x tick value; USDJPY tick value ~0.636 USD at 157).
Commission $7/lot round turn for every symbol (conservative). Slippage per side in points.
News: USD high-impact for all symbols, plus EUR (ECB, EU CPI, DE GDP) for EURUSD / GER40 and JPY (BoJ, JP data)
for USDJPY. Indices (".r" CFDs): contract 1, lots 0.1 step 0.1, no commission (FundingPips charges none on indices),
swaps in USD (GER40 in EUR, converted), triple swap on Friday (irrelevant: we are flat every weekend).

News rule (docs/rules.md, 2026-09-27): no new position from 10 min before to 10 min after a high-impact event;
positions opened < 5 h before the event are closed 10 min before it (FundingPips funded rule: profits of trades
opened or closed within +-5 min are deducted unless the trade was opened >= 5 h before the event).
"""
import numpy as np
import pandas as pd

from research import calendar_news, engine, lab

SPECS = {
    "XAUUSD": dict(point=0.01, contract=100.0, quote="USD", news=("usd",)),
    "XAGUSD": dict(point=0.001, contract=5000.0, quote="USD", news=("usd",)),
    "EURUSD": dict(point=0.00001, contract=100_000.0, quote="USD", news=("usd", "eur")),
    "USDJPY": dict(point=0.001, contract=100_000.0, quote="JPY", news=("usd", "jpy")),
    "GBPUSD": dict(point=0.00001, contract=100_000.0, quote="USD", news=("usd",)),
    "AUDUSD": dict(point=0.00001, contract=100_000.0, quote="USD", news=("usd",)),
    "NZDUSD": dict(point=0.00001, contract=100_000.0, quote="USD", news=("usd",)),
    "USDCHF": dict(point=0.00001, contract=100_000.0, quote="CHF", news=("usd",)),
    "USDCAD": dict(point=0.00001, contract=100_000.0, quote="CAD", news=("usd",)),
    "NAS100": dict(point=0.01, contract=1.0, quote="USD", news=("usd",), vmin=0.1, vstep=0.1, file="NAS100"),
    "DJ30": dict(point=0.01, contract=1.0, quote="USD", news=("usd",), vmin=0.1, vstep=0.1, file="DJ30"),
    "GER40": dict(point=0.01, contract=1.0, quote="EUR", news=("usd", "eur"), vmin=0.1, vstep=0.1, file="GER40"),
    "SP500": dict(point=0.01, contract=1.0, quote="USD", news=("usd",), vmin=0.1, vstep=0.1, file="SP500"),
    # FundingPips FTSE100 / JP225 (broker: UK100.r cash 2021+, JPN225ft future 2022-12+); fx = (pair, invert) -> USD per quote unit
    "UK100": dict(point=0.01, contract=1.0, quote="GBP", fx=("GBPUSD", False), news=("usd",), vmin=0.1, vstep=0.1, file="UK100"),
    "JP225": dict(point=0.01, contract=1.0, quote="JPY", fx=("USDJPY", True), news=("usd", "jpy"), vmin=1.0, vstep=1.0, file="JPN225"),
}
NEWS_BEFORE, NEWS_AFTER, NEWS_FLATTEN, NEWS_EXEMPT_MIN = 10, 10, 10, 290
COSTS = {
    "XAUUSD": engine.Costs(),
    "XAGUSD": engine.Costs(min_spread_pts=15.0, slippage_pts=5.0, swap_long=-119.15, swap_short=10.70),
    "EURUSD": engine.Costs(min_spread_pts=5.0, slippage_pts=3.0, swap_long=-5.76, swap_short=2.50),
    "USDJPY": engine.Costs(min_spread_pts=5.0, slippage_pts=3.0, swap_long=4.30, swap_short=-13.68),
    "NAS100": engine.Costs(commission_per_lot=0.0, min_spread_pts=50.0, slippage_pts=50.0, swap_long=-6.08,
                           swap_short=1.12, triple_dow=4),
    "DJ30": engine.Costs(commission_per_lot=0.0, min_spread_pts=100.0, slippage_pts=100.0, swap_long=-10.61,
                         swap_short=1.96, triple_dow=4),
    "GER40": engine.Costs(commission_per_lot=0.0, min_spread_pts=50.0, slippage_pts=50.0, swap_long=-4.30,
                          swap_short=0.37, triple_dow=4),
    "UK100": engine.Costs(commission_per_lot=0.0, min_spread_pts=80.0, slippage_pts=50.0, swap_long=-2.15,
                          swap_short=0.40, triple_dow=4),
    "JP225": engine.Costs(commission_per_lot=0.0, min_spread_pts=1000.0, slippage_pts=500.0, swap_long=0.0,
                          swap_short=0.0, triple_dow=4),
    "SP500": engine.Costs(commission_per_lot=0.0, min_spread_pts=30.0, slippage_pts=25.0, swap_long=-1.57,
                          swap_short=0.29, triple_dow=4),
}
CAL = {"usd": "calendar_usd_high.csv", "eur": "calendar_eur_high.csv", "jpy": "calendar_jpy_high.csv"}


def load_m1(sym):
    """M1 bars; early index history has no recorded spread (0) -> filled with the symbol's median recorded spread
    in bps of price (NAS100/DJ30 before 2020, GER40 before 2024)."""
    d = pd.read_parquet(lab.DATA / f"{SPECS[sym].get('file', sym)}_M1_2018.parquet")
    zero = d["spread"] <= 0
    if zero.mean() > 0.05:
        pt = SPECS[sym]["point"]
        bps = (d.loc[~zero, "spread"] * pt / d.loc[~zero, "close"]).median()
        d.loc[zero, "spread"] = (bps * d.loc[zero, "close"] / pt).round()
    return d


def news_times(sym):
    parts = [calendar_news.load_news_server_times(lab.DATA / CAL[k]) for k in SPECS[sym]["news"]]
    return pd.DatetimeIndex(np.unique(np.concatenate([p.values for p in parts])))


def prepare(sym, bars, before=NEWS_BEFORE, after=NEWS_AFTER, flatten=NEWS_FLATTEN, costs=None):
    news = news_times(sym)
    blk, flt = calendar_news.blackout_masks(bars.index, 1, news, before, after, flatten)
    sp = SPECS[sym]
    fx = load_m1("EURUSD")["close"] if sp["quote"] == "EUR" else None
    if "fx" in sp:
        px = load_m1(sp["fx"][0])["close"]
        fx = 1.0 / px if sp["fx"][1] else px
    return engine.prepare_exec(bars, 1, blk, flt, costs or COSTS[sym], point=sp["point"], contract=sp["contract"],
                               quote=sp["quote"], fx=fx, vmin=sp.get("vmin", 0.01), vstep=sp.get("vstep", 0.01))

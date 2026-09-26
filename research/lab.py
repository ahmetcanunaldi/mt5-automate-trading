"""Experiment harness: data loading, period splits, evaluation, logging.

Periods (constrained by what the terminal can serve today, see docs/knowhow.md):
  DEV : 2022-07-05 .. 2024-11-30  signals M15, execution on M15 bars (pessimistic intrabar)
  VAL : 2024-12-04 .. 2025-09-30  execution on tick-built M1 bars
  OOS : 2025-10-01 .. 2026-09-25  execution on tick-built M1 bars  -- LOCKED, final candidates only
Every period starts from a fresh $10,000 account.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, metrics  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

PERIODS = {
    "DEV": ("2022-07-05", "2024-11-30", "M15"),
    "VAL": ("2024-12-04", "2025-09-30", "M1"),
    "OOS": ("2025-10-01", "2026-09-26", "M1"),
    # tick-era split for M5-signal research (both inside the old VAL window; OOS untouched)
    "TDEV": ("2024-12-04", "2025-06-30", "M1"),
    "TVAL": ("2025-07-01", "2025-09-30", "M1"),
}
_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = pd.read_parquet(DATA / f"XAUUSD_{name}.parquet")
    return _cache[name]


def signal_bars(tf="M15"):
    """Continuous signal-timeframe bars (terminal M15 covers 2022-07 onwards)."""
    if tf == "M15":
        return load("M15")
    if tf == "H1":
        return load("H1")
    if tf == "M5":
        m1 = load("M1_from_ticks")
        return m1.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                         "tick_volume": "sum", "spread": "mean"}).dropna(subset=["open"])
    raise ValueError(tf)


def exec_data(period, news_cfg=(30, 30, 10), costs=engine.Costs()):
    a, b, tf = PERIODS[period]
    key = ("exec", period, news_cfg, tuple(sorted(vars(costs).items())))
    if key in _cache:
        return _cache[key]
    if tf == "M15":
        bars, bm = load("M15").loc[a:b], 15
    elif tf == "M1L":                      # long tester export 2018-09..
        bars, bm = load("M1_2018").loc[a:b], 1
    else:
        bars, bm = load("M1_from_ticks").loc[a:b], 1
    news = calendar_news.load_news_server_times()
    block, flat = calendar_news.blackout_masks(bars.index, bm, news, *news_cfg)
    x = engine.prepare_exec(bars, bm, block, flat, costs)
    _cache[key] = x
    return x


def evaluate(signals, period, guards=engine.Guards(), costs=engine.Costs(), label="", with_challenge=True):
    if period == "OOS" and not getattr(evaluate, "oos_unlocked", False):
        raise PermissionError("OOS is locked. Set lab.evaluate.oos_unlocked = True only for a logged final candidate.")
    a, b, _ = PERIODS[period]
    x = exec_data(period, costs=costs)
    s = signals.loc[a:b]
    res = engine.run(x, s, guards, costs)
    m = metrics.summarize(res, f"{label}|{period}")
    m.update(metrics.monte_carlo_dd(res))
    if with_challenge:
        m["challenge"] = metrics.challenge_sim(res)
    m["gates"] = metrics.gate_check(m)
    return m, res


def save_experiment(exp_id, meta: dict, results: dict, res_objs: dict | None = None):
    d = REPORTS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=str),
                                    encoding="utf-8")
    if res_objs:
        from research.plots import equity_report
        for per, r in res_objs.items():
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in per)
            r.trades.to_csv(d / f"trades_{safe}.csv", index=False)
            m = results.get(per, {})
            ttl = (f"{exp_id} {per} | n={m.get('trades')} SR={m.get('sharpe')} PF={m.get('profit_factor')} "
                   f"net={m.get('net_pct')}% DD={m.get('max_total_dd_pct')}% dDD={m.get('max_daily_dd_pct')}%")
            equity_report(r, ttl, d / f"equity_{safe}.png")


def fmt(m):
    c = m.get("challenge", {})
    return (f"{m['label']:<40} n={m['trades']:>4} wr={m['win_rate']:.2f} PF={m['profit_factor']:.2f} "
            f"net={m['net_pct']:>6.1f}% SR={m['sharpe']:>5.2f} DD={m['max_total_dd_pct']:.1f}% "
            f"dDD={m['max_daily_dd_pct']:.1f}% mc95={m['mc_dd95_pct']} pass={c.get('pass_rate')} "
            f"fail={c.get('fail_rate')} med_days={c.get('median_days_to_pass')}")

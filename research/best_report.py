"""EXP-069: full report of the current best book = best12 + season(0.5x) + trailing 1.5x on intraday legs."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import B  # noqa: E402
from research.combo8 import BEST12, build  # noqa: E402
from research.exits import INTRADAY  # noqa: E402
from research.final_candidate import g  # noqa: E402


def book_signals(legs, start):
    parts = []
    for n in BEST12 + ["season"]:
        s = legs[n].loc[start:].copy()
        if n == "season":
            s["risk_mult"] = 0.5
        if n in INTRADAY:
            s["trail"] = 1.5 * s.sl
        parts.append(s)
    return pd.concat(parts).sort_index(kind="stable")


if __name__ == "__main__":
    m1, legs = build()
    news = calendar_news.load_news_server_times()
    results, objs = {}, {}
    for a in ("2019-01-01", "2021-01-01", "2025-01-01"):
        ex = m1.loc[a:B]
        blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
        res = engine.run(engine.prepare_exec(ex, 1, blk, flt), book_signals(legs, a), g(True, 6))
        m = metrics.summarize(res, f"best13 from {a[:4]}"); m.update(metrics.monte_carlo_dd(res))
        m["challenge"] = metrics.challenge_sim(res, max_days=250); m["gates"] = metrics.gate_check(m)
        eq = res.daily["end"]
        yrs = eq.resample("YE").last(); ypnl = yrs.diff().fillna(yrs.iloc[0] - 100_000)
        mon = eq.resample("ME").last(); mp = mon.diff().fillna(mon.iloc[0] - 100_000)
        print(f"\n=== from {a[:4]} === final ${eq.iloc[-1]:,.0f} ({m['net_pct']:+.1f} %), CAGR {m['cagr_pct']} %, SR {m['sharpe']}, "
              f"peak DD {m['max_total_dd_pct']} %, worst day {m['max_daily_dd_pct']} %, MC95 {m['mc_dd95_pct']} %")
        print("   P&L by year:", {k.year: round(v) for k, v in ypnl.items()})
        print(f"   months: mean ${mp.mean():,.0f}, >= 2 %: {(mp >= 2000).mean() * 100:.0f} %, positive {(mp > 0).mean() * 100:.0f} %")
        print(f"   weeks: {m['week_R_mean']} R mean, median {m['week_R_median']} R, >= 2R {m['weeks_ge_2R'] * 100:.0f} %, positive {m['weeks_pos'] * 100:.0f} %")
        print("   challenge:", m["challenge"], "\n   gates:", m["gates"])
        key = f"from_{a[:4]}"; results[key] = m; objs[key] = res
    lab.save_experiment("EXP-069", {"book": "best12 + season0.5 + trail1.5x intraday"}, results, objs)

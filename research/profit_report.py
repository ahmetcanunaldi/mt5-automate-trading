"""EXP-054: how much does the best candidate make? $100k, 0.5 % risk, 10 legs (EXP-047 + tday900), weekend flat,
K6, real guards. Yearly / monthly / weekly P&L, weekly R distribution vs the new 2 R/week target."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.final_candidate import LEGS9, g  # noqa: E402
from research.run_trend_intraday import scan, setup as td_setup  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402

if __name__ == "__main__":
    m1, legs = all_legs()
    m15_td, day, _ = td_setup()
    s = scan(m15_td, day, "tday", 0.3, 1.0, None, 900); s["hold_min"] = 510; s["leg"] = "tday900"
    legs["tday900"] = s.loc[A:B]
    names = LEGS9 + ["tday900"]
    news = calendar_news.load_news_server_times()
    results, objs = {}, {}
    for a in ("2019-01-01", "2021-01-01", "2025-01-01"):
        ex = m1.loc[a:B]
        blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
        x = engine.prepare_exec(ex, 1, blk, flt)
        sig = pd.concat([legs[n].loc[a:] for n in names]).sort_index(kind="stable")
        res = engine.run(x, sig, g(True, 6))
        m = metrics.summarize(res, f"best10|from {a[:4]}"); m.update(metrics.monte_carlo_dd(res))
        m["challenge"] = metrics.challenge_sim(res, max_days=250); m["gates"] = metrics.gate_check(m)
        key = f"from_{a[:4]}"
        results[key] = m; objs[key] = res
        eq = res.daily["end"]
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        yearly = eq.resample("YE").last()
        y_pnl = yearly.diff().fillna(yearly.iloc[0] - 100_000)
        mon = eq.resample("ME").last(); m_pnl = mon.diff().fillna(mon.iloc[0] - 100_000)
        wk = eq.resample("W-FRI").last(); w_pnl = wk.diff().fillna(wk.iloc[0] - 100_000)
        print(f"\n=== start {a[:10]} -> 2026-09-25 ({yrs:.1f} y) ===")
        print(f"final equity ${eq.iloc[-1]:,.0f}  total {m['net_pct']:+.1f} %  CAGR {m['cagr_pct']:.1f} %  Sharpe {m['sharpe']}  "
              f"peak DD {m['max_total_dd_pct']} %  worst day {m['max_daily_dd_pct']} %")
        print("P&L by year ($):", {k.year: round(v) for k, v in y_pnl.items()})
        print(f"months: mean ${m_pnl.mean():,.0f}, median ${m_pnl.median():,.0f}, positive {(m_pnl > 0).mean() * 100:.0f} %, "
              f"best ${m_pnl.max():,.0f}, worst ${m_pnl.min():,.0f}")
        print(f"weeks : mean ${w_pnl.mean():,.0f} = {m['week_R_mean']} R, median {m['week_R_median']} R, "
              f"weeks >= 2R {m['weeks_ge_2R'] * 100:.0f} %, weeks > 0 {m['weeks_pos'] * 100:.0f} %")
        print("challenge (P1+P2 within 250 d):", m["challenge"])
        print("gates:", m["gates"])
    lab.save_experiment("EXP-054", {"legs": names}, results, objs)

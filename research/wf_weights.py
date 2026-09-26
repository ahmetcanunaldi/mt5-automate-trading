"""EXP-045: walk-forward risk weighting of legs (never above 0.5 % per trade).

For each year Y in 2021..2026 every leg gets weight w = clip(t_trailing / 3, 0, 1) where t_trailing is the t-stat
of its per-trade R over the two previous calendar years (single-leg runs); w = 0 if t <= 0.5. The portfolio for
2021-2026 is then run ONCE with the engine (K6, open risk <= 3 %, real guards) using risk_mult = w of the leg in
that year. Compared with equal weights (all positive-family legs) on the same window.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import A, B, build_legs  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_volbreak import daily_ctx, signals as vb_signals  # noqa: E402
from research.strategies import swing  # noqa: E402
from research.strategies.amd import amd_signals  # noqa: E402

SWING = {"trendH4", "tom", "h4_rsi2"}


def all_legs():
    m1, legs = build_legs()
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    h4 = m1.resample("4h").agg(AGG).dropna(subset=["open"])
    ctx = daily_ctx(m1)
    legs["lw"] = vb_signals(m15.loc[A:B], ctx, "lw", k=0.4, sl=1.0, start_min=600)
    legs["inside"] = vb_signals(m15.loc[A:B], ctx, "nr", pattern="inside", trend=True)
    legs["nr7"] = vb_signals(m15.loc[A:B], ctx, "nr", pattern="nr7", trend=True)
    a15 = atr(m15, 14)
    days = pd.DatetimeIndex(np.unique(m1.loc[A:B].index.normalize()))
    fri = days[days.dayofweek == 4] + pd.Timedelta(minutes=1385)
    legs["fri_close"] = pd.DataFrame({"dir": 1, "sl": 4.0 * a15.asof(fri - pd.Timedelta(minutes=15)).to_numpy(),
                                      "tp": 1e6, "hold_min": 50, "be": 0.0, "trail": 0.0}, index=fri).dropna()
    legs["h4_rsi2"] = swing.rsi2(swing.prep(h4, bar=pd.Timedelta(hours=4)), lo=10, hi=90, sl=2.0, tp=1.0, hold=6)
    legs["amd_lny"] = amd_signals(m15, "london_ny", sweep_atr=0.5, confirm_bars=2, tp_mode="range")
    for k in list(legs):
        legs[k] = legs[k].loc[A:B].copy(); legs[k]["leg"] = k
    return m1, legs


def guards_for(leg, K=1, open_risk=0.5):
    fc = leg == "fri_close"
    return engine.Guards(initial_balance=100_000, intraday=leg not in SWING and leg != "portfolio", weekend_flat=False,
                         last_entry_min=1390, flatten_min=1437 if fc else 1425, fri_flatten_min=1500 if fc else 1350,
                         max_trades_day=8, max_positions=K, max_open_risk_pct=open_risk,
                         total_stop_pct=100.0 if leg != "portfolio" else 8.0,
                         total_derisk_pct=100.0 if leg != "portfolio" else 6.5)


if __name__ == "__main__":
    m1, legs = all_legs()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    stats = {}
    for k, s in legs.items():
        t = engine.run(x, s, guards_for(k)).trades
        stats[k] = t.assign(y=t.entry_time.dt.year)[["y", "R"]]
    W = {}
    for y in range(2021, 2027):
        W[y] = {}
        for k, t in stats.items():
            r = t[(t.y >= y - 2) & (t.y <= y - 1)].R
            tt = r.mean() / r.std() * np.sqrt(len(r)) if len(r) > 10 and r.std() > 0 else 0.0
            W[y][k] = 0.0 if tt <= 0.5 else float(min(1.0, tt / 3.0))
    Wd = pd.DataFrame(W).round(2)
    print("walk-forward weights (rows=leg, cols=year):\n", Wd.to_string())
    x21 = engine.prepare_exec(ex.loc["2021":], 1, blk[ex.index >= "2021-01-01"], flt[ex.index >= "2021-01-01"])
    positive = ["trendH4", "tday", "drift", "friday", "tom", "lw", "inside", "nr7", "fri_close"]
    books = {}
    parts = []
    for k, s in legs.items():
        s = s.loc["2021":].copy()
        s["risk_mult"] = [W[ts.year][k] for ts in s.index]
        parts.append(s[s.risk_mult > 0])
    books["WF_weighted"] = pd.concat(parts).sort_index(kind="stable")
    books["equal_positive"] = pd.concat([legs[k].loc["2021":] for k in positive]).sort_index(kind="stable")
    rows, keep = [], {}
    for name, s in books.items():
        res = engine.run(x21, s, guards_for("portfolio", K=6, open_risk=3.0))
        t = res.trades
        m = metrics.summarize(res, name); m.update(metrics.monte_carlo_dd(res))
        ch = metrics.challenge_sim(res, max_days=250)
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        rows.append({"book": name, "n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / 5.7, 1), "SR": m["sharpe"],
                     "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
                     "mc95": m["mc_dd95_pct"], "pass250": ch["pass_rate"], "med": ch["median_days_to_pass"],
                     **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        m["challenge"] = ch; m["gates"] = metrics.gate_check(m)
        keep[name] = (m, res)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    (lab.REPORTS / "EXP-045").mkdir(exist_ok=True)
    Wd.to_csv(lab.REPORTS / "EXP-045" / "weights.csv")
    lab.save_experiment("EXP-045", {"weights": Wd.to_dict()}, {k: v[0] for k, v in keep.items()},
                        {k: v[1] for k, v in keep.items()})

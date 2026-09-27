"""EXP-099: how much of v2 is timing and how much is just being long a rising market?
For every long-only leg, a benchmark with the SAME entry time, hold, stop and trail but taken EVERY eligible day
(no condition). Excess = leg avg R - benchmark avg R; Welch t of the difference. Directional legs (lw, tday,
tday900, inside, nr7) are split into long/short trades (shorts cannot be drift). Also the benchmark portfolio
(always-long XAU + NAS100 + DJ30 at the same risk) vs v2 on Sharpe."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, engine_multi, lab, legcache, metrics, symbols  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.portfolio_v2 import book, guards, load_all  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def g1():
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                         max_trades_day=8, max_positions=1, max_open_risk_pct=0.5, risk_on_initial=True,
                         total_stop_pct=100.0, total_derisk_pct=100.0)


def every_day(m1, t_min, hold, sl_k, trail_k, dows=(0, 1, 2, 3, 4), atr_tf="D"):
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    days = d1.loc[A:B].index; days = days[days.dayofweek.isin(dows)]
    t = days + pd.Timedelta(minutes=t_min)
    if atr_tf == "D":
        a = atr(d1, 14).shift(1).reindex(days).to_numpy()
    else:
        h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"]); a = atr(h1, 14).asof(t - pd.Timedelta(hours=1)).to_numpy()
    s = pd.DataFrame({"dir": 1, "sl": sl_k * a, "tp": 1e6, "hold_min": hold, "be": 0.0, "trail": trail_k * a, "leg": "bench"}, index=t)
    return s[np.isfinite(s.sl) & (s.sl > 0)]


def run1(x, s, sym):
    return engine.run(x, s, g1(), symbols.COSTS[sym]).trades


def cmp(label, leg, bench):
    d = leg.R.mean() - bench.R.mean()
    t = d / np.sqrt(leg.R.var() / len(leg) + bench.R.var() / len(bench))
    return {"leg": label, "n": len(leg), "leg_avgR": round(leg.R.mean(), 3), "bench_n": len(bench),
            "bench_avgR": round(bench.R.mean(), 3), "excess_R": round(d, 3), "t_excess": round(t, 2)}


if __name__ == "__main__":
    m1x, legs = legcache.load()
    xx = symbols.prepare("XAUUSD", m1x.loc[A:B])
    rows = []
    B_day = run1(xx, every_day(m1x, 65, 1360, 1.0, 1.5), "XAUUSD")
    for name in ("strong_close", "season"):
        rows.append(cmp(f"XAU {name}", run1(xx, book_signals(legs, A).query("leg == @name"), "XAUUSD"), B_day))
    B_fri = run1(xx, every_day(m1x, 65, 1315, 1.5, 2.25), "XAUUSD")
    rows.append(cmp("XAU friday (bench: every day same rule)", run1(xx, book_signals(legs, A).query("leg == 'friday'"), "XAUUSD"), B_fri))
    B_dr = run1(xx, every_day(m1x, 75, 475, 4.0, 6.0, dows=(1, 2, 3, 4), atr_tf="H"), "XAUUSD")
    B_dr_all = run1(xx, every_day(m1x, 75, 475, 4.0, 6.0, dows=(0, 1, 2, 3, 4), atr_tf="H"), "XAUUSD")
    B_any = run1(xx, every_day(m1x, 600, 660, 4.0, 6.0, dows=(0, 1, 2, 3, 4), atr_tf="H"), "XAUUSD")
    rows.append(cmp("XAU drift (bench: Tue-Fri 01:15 w/o vol filter)", run1(xx, book_signals(legs, A).query("leg == 'drift'"), "XAUUSD"), B_dr))
    rows.append(cmp("XAU Asia 01:15 hold (bench: 10:00 same hold)", B_dr_all, B_any))
    tr_all = run1(xx, book_signals(legs, A).query("leg == 'trendH4'"), "XAUUSD")
    rows.append(cmp("XAU trendH4 (bench: every day long)", tr_all, B_day))
    for name in ("lw", "tday", "tday900", "inside", "nr7"):
        t = run1(xx, book_signals(legs, A).query("leg == @name"), "XAUUSD")
        for dd, tag in ((1, "long"), (-1, "short")):
            sub = t[t.dir == dd]
            if len(sub) > 5:
                rows.append({"leg": f"XAU {name} {tag} trades (bench: 0)", "n": len(sub), "leg_avgR": round(sub.R.mean(), 3),
                             "bench_n": 0, "bench_avgR": 0.0, "excess_R": round(sub.R.mean(), 3),
                             "t_excess": round(sub.R.mean() / sub.R.std() * np.sqrt(len(sub)), 2)})
    for sym in ("NAS100", "DJ30"):
        m1 = symbols.load_m1(sym); x = symbols.prepare(sym, m1.loc[A:B])
        L = pd.read_pickle(lab.DATA / f"legs_{sym}.pkl")
        Bd = run1(x, every_day(m1, 65, 1345, 1.0, 1.5), sym)
        Bm = run1(x, every_day(m1, 65, 1345, 1.5, 2.25), sym)
        for name in ("mon", "dip_low20", "dip_clv", "hi20", "prefomc"):
            rows.append(cmp(f"{sym} {name}", run1(x, L[name], sym), Bm if name in ("mon", "prefomc") else Bd))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    # portfolio: always-long benchmark vs v2
    ax, xau, idx_legs = load_all()
    costs = {s: symbols.COSTS[s] for s in ["XAUUSD", "NAS100", "DJ30", "GER40"]}
    bench = pd.concat([every_day(m1x, 65, 1360, 1.0, 1.5).assign(sym="XAUUSD"),
                       every_day(symbols.load_m1("NAS100"), 66, 1344, 1.0, 1.5).assign(sym="NAS100"),
                       every_day(symbols.load_m1("DJ30"), 67, 1343, 1.0, 1.5).assign(sym="DJ30")]).sort_index(kind="stable")
    for name, sig in (("always-long XAU+NAS+DJ (1 pos each, every day)", bench),
                      ("v2", book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR}))):
        res = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True))
        m = metrics.summarize(res, name)
        print(f"{name}: SR {m['sharpe']}, trades {m['trades']}, net {m['net_pct']} %, DD {m['max_total_dd_pct']} %, week R {m['week_R_mean']}")
    (lab.REPORTS / "EXP-099").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-099" / "beta_vs_timing.csv", index=False)

"""EXP-102: index expansion in our own style — the same literature-prior legs (mon, dip_low20, dip_clv, hi20, prefomc,
tom) on the FundingPips indices not yet in the book: SP500 (SPX500), UK100 (FTSE100, broker history 2021+),
JP225 (broker future 2022-12+), GER40 (re-check). No new leg mining, no parameter changes.
A. standalone per leg (fixed $500, index guards): n, avg R, t, t by era, positive years.
B. portfolio: v2.2 (XAU best13 + NAS100/DJ30 prior legs, K6, open risk 3 %, cushion 2 %) + each new index's prior
   legs, and + all; K 6 / 8; challenge + funded metrics; improvement split by era (2019-22 vs 2023-26)."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, engine_multi, lab, legcache, metrics, symbols  # noqa: E402
from research import index_legs as IL  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.multi_scan import tstat  # noqa: E402
from research.payout_sim import summarize_payouts  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
NEW = ["SP500", "UK100", "JP225", "GER40"]
BASE = {"NAS100": PRIOR, "DJ30": PRIOR}
for s in ("SP500", "UK100", "JP225"):
    IL.SESSION[s] = (65, 1410)
OUT = lab.REPORTS / "EXP-102"


def standalone():
    rows, legs = [], {}
    for sym in NEW:
        m1 = symbols.load_m1(sym)
        x = symbols.prepare(sym, m1.loc[A:B])
        L = IL.index_legs(m1, sym)
        legs[sym] = L
        pd.to_pickle(L, lab.DATA / f"legs_{sym}.pkl")
        yrs = (m1.index[-1] - max(m1.index[0], pd.Timestamp(A))).days / 365.25
        for name in PRIOR:
            t = engine.run(x, L[name], IL.gidx(), symbols.COSTS[sym]).trades
            if len(t) < 10:
                continue
            yr = t.groupby(t.entry_time.dt.year).R.sum()
            mid = t.entry_time.quantile(0.5) if len(t) else None
            rows.append({"sym": sym, "leg": name, "n": len(t), "avgR": round(t.R.mean(), 3), "t": tstat(t.R),
                         "R_yr": round(t.R.sum() / yrs, 1), "t_first_half": tstat(t[t.entry_time <= mid].R),
                         "t_second_half": tstat(t[t.entry_time > mid].R), "yrs_pos": f"{int((yr > 0).sum())}/{len(yr)}"})
        book = pd.concat([L[n] for n in PRIOR]).sort_index(kind="stable")
        t = engine.run(x, book, IL.gidx(), symbols.COSTS[sym]).trades
        rows.append({"sym": sym, "leg": "ALL prior", "n": len(t), "avgR": round(t.R.mean(), 3), "t": tstat(t.R),
                     "R_yr": round(t.R.sum() / yrs, 1), "yrs_pos": f"{int((t.groupby(t.entry_time.dt.year).R.sum() > 0).sum())}"})
    return pd.DataFrame(rows), legs


def run_book(ax, sig, syms, K, orisk, label):
    costs = {s: symbols.COSTS[s] for s in syms}
    gd = dataclasses.replace(g(True, K), max_open_risk_pct=orisk, cushion_start_pct=2.0)
    res = engine_multi.run(ax, sig, costs, gd)
    m = metrics.summarize(res, label); m.update(metrics.monte_carlo_dd(res))
    ch = metrics.challenge_sim(res, max_days=250)
    d = res.daily["end"]; r = d.pct_change().fillna(0)
    sr = lambda x: round(float(x.mean() / x.std() * np.sqrt(252)), 2)  # noqa: E731
    rf = engine_multi.run(ax, sig, costs, dataclasses.replace(gd, risk_on_initial=True, payout_pct=3.0, consistency_pct=35.0,
                                                              day_profit_cap_pct=1.25))
    pr, _ = summarize_payouts(rf, "")
    return {"book": label, "K": K, "orisk": orisk, "n": len(res.trades), "SR": m["sharpe"], "CAGR%": m["cagr_pct"],
            "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"], "weekR": m["week_R_mean"],
            "SR_19_22": sr(r[r.index.year <= 2022]), "SR_23_26": sr(r[r.index.year >= 2023]),
            "pass250": ch["pass_rate"], "fail": ch["fail_rate"], "med_pass_d": ch["median_days_to_pass"],
            "payouts": pr["payouts"], "withdrawn$": pr.get("withdrawn_$"), "pay_med_d": pr.get("median_days")}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    T, legs = standalone()
    pd.set_option("display.width", 250)
    print(T.to_string(index=False)); T.to_csv(OUT / "standalone.csv", index=False)
    m1x, xl = legcache.load()
    syms = ["XAUUSD", "NAS100", "DJ30"] + NEW
    execs = {"XAUUSD": symbols.prepare("XAUUSD", m1x.loc[A:B])}
    for s in syms[1:]:
        execs[s] = symbols.prepare(s, symbols.load_m1(s).loc[A:B])
    ax = engine_multi.align(execs)
    idx = {s: pd.read_pickle(lab.DATA / f"legs_{s}.pkl") for s in syms[1:]}
    xau = book_signals(xl, A).assign(sym="XAUUSD")

    def book(sets):
        parts = [xau] + [idx[s][n].assign(sym=s, leg=f"{s}:{n}") for s, names in sets.items() for n in names]
        return pd.concat(parts).sort_index(kind="stable")

    variants = {"v2.2": BASE}
    for s in NEW:
        variants[f"v2.2+{s}"] = {**BASE, s: PRIOR}
    variants["v2.2+SP500+UK100+JP225"] = {**BASE, "SP500": PRIOR, "UK100": PRIOR, "JP225": PRIOR}
    rows = []
    for name, sets in variants.items():
        for K, orisk in ((6, 3.0), (8, 4.0)):
            if name == "v2.2" and K == 8:
                continue
            row = run_book(ax, book(sets), syms, K, orisk, name)
            rows.append(row); print(row, flush=True)
    P = pd.DataFrame(rows); P.to_csv(OUT / "portfolio.csv", index=False)
    print(P.to_string(index=False))

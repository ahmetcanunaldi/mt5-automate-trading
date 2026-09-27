"""EXP-085 (approximation): XAUUSD best13 + index books, each run separately at fixed $500 risk (risk on initial),
daily P&L summed. Two index leg sets to expose selection bias:
  PRIOR    : the same literature-prior legs on all three indices (mon, dip_low20, dip_clv, hi20, prefomc, tom)
  SELECTED : per-index legs with t >= 2 in EXP-084 (NAS: mon, hi20, dip_low20, sw_tsmom120_L, xau_tday,
             sw_ema20_100_L, prefomc; DJ30: dip_low20, dip_clv, mon; GER40: none)
Metrics on the summed daily P&L: Sharpe, R/yr, max DD (peak-to-trough of cumulative P&L / $100k), worst day,
weekly R, correlation between books."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, legcache, symbols  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.final_candidate import g  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
PRIOR = ["mon", "dip_low20", "dip_clv", "hi20", "prefomc", "tom"]
SELECTED = {"NAS100": ["mon", "hi20", "dip_low20", "sw_tsmom120_L", "xau_tday", "sw_ema20_100_L", "prefomc"],
            "DJ30": ["dip_low20", "dip_clv", "mon"], "GER40": []}


def fixed(K, orisk):
    return dataclasses.replace(g(True, K), max_open_risk_pct=orisk, risk_on_initial=True)


def daily_pnl(res):
    eq = res.daily["end"]
    return eq.diff().fillna(eq.iloc[0] - 100_000)


def stats(p, label):
    p = p[p.index.dayofweek < 5]
    cum = p.cumsum()
    dd = (np.maximum.accumulate(np.maximum(cum, 0)) - cum).max() / 1000
    wk = p.resample("W-FRI").sum() / 500
    return {"book": label, "SR": round(p.mean() / p.std() * np.sqrt(252), 2), "R_yr": round(p.sum() / 500 / 7.7, 1),
            "maxDD%": round(dd, 2), "worst_day%": round(p.min() / 1000, 2), "week_R": round(wk.mean(), 2),
            "wk>=2R": round((wk >= 2).mean(), 3), "yrs_pos": int((p.groupby(p.index.year).sum() > 0).sum())}


if __name__ == "__main__":
    m1, legs = legcache.load()
    xau = engine.run(symbols.prepare("XAUUSD", m1.loc[A:B]), book_signals(legs, A), fixed(6, 3.0))
    books = {"XAU": daily_pnl(xau)}
    for sym in ("NAS100", "DJ30", "GER40"):
        L = pd.read_pickle(lab.DATA / f"legs_{sym}.pkl")
        x = symbols.prepare(sym, symbols.load_m1(sym).loc[A:B])
        for tag, names in (("prior", PRIOR), ("sel", SELECTED[sym])):
            if not names:
                continue
            s = pd.concat([L[n] for n in names]).sort_index(kind="stable")
            res = engine.run(x, s, fixed(4, 2.0), symbols.COSTS[sym])
            books[f"{sym}_{tag}"] = daily_pnl(res)
    P = pd.DataFrame(books).fillna(0.0)
    rows = [stats(P[c], c) for c in P]
    prior = [c for c in P if c.endswith("prior")]; sel = [c for c in P if c.endswith("sel")]
    rows.append(stats(P[prior].sum(axis=1), "indices prior (3)"))
    rows.append(stats(P[sel].sum(axis=1), "indices selected"))
    rows.append(stats(P[["XAU"] + prior].sum(axis=1), "XAU + indices prior"))
    rows.append(stats(P[["XAU"] + sel].sum(axis=1), "XAU + indices selected"))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    print("\ndaily P&L correlation:\n", P[P.index.dayofweek < 5].corr().round(2).to_string())
    (lab.REPORTS / "EXP-085").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-085" / "summary.csv", index=False)
    P.to_csv(lab.REPORTS / "EXP-085" / "daily_pnl.csv")

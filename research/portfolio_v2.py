"""EXP-086: one account, XAUUSD best13 + index legs, shared guards (multi-symbol engine), new news rule.
Index leg sets: PRIOR (same literature legs on each index), PRIOR on NAS100+DJ30 only, SELECTED (EXP-084 t >= 2).
Capacity grid: total positions K and open-risk cap. Challenge mode (compounding) and funded mode (fixed $500,
+3 % payouts, 35 % consistency, day cap)."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import engine_multi, lab, legcache, metrics, symbols  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo_idx import PRIOR, SELECTED  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.payout_sim import summarize_payouts  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
SYMS = ["XAUUSD", "NAS100", "DJ30", "GER40"]


def load_all():
    m1, legs = legcache.load()
    execs = {"XAUUSD": symbols.prepare("XAUUSD", m1.loc[A:B])}
    for s in SYMS[1:]:
        execs[s] = symbols.prepare(s, symbols.load_m1(s).loc[A:B])
    ax = engine_multi.align(execs)
    idx_legs = {s: pd.read_pickle(lab.DATA / f"legs_{s}.pkl") for s in SYMS[1:]}
    return ax, book_signals(legs, A).assign(sym="XAUUSD"), idx_legs


def book(xau, idx_legs, sets):
    parts = [xau]
    for s, names in sets.items():
        for n in names:
            parts.append(idx_legs[s][n].assign(sym=s, leg=f"{s}:{n}"))
    return pd.concat(parts).sort_index(kind="stable")


def guards(K, orisk, **kw):
    return dataclasses.replace(g(True, K), max_open_risk_pct=orisk, **kw)


def evaluate(ax, sig, K, orisk, label, funded=True):
    costs = {s: symbols.COSTS[s] for s in SYMS}
    res = engine_multi.run(ax, sig, costs, guards(K, orisk))
    m = metrics.summarize(res, label); m.update(metrics.monte_carlo_dd(res))
    ch = metrics.challenge_sim(res, max_days=250)
    t = res.trades
    row = {"book": label, "K": K, "orisk": orisk, "n": len(t), "SR": m["sharpe"], "CAGR%": m["cagr_pct"],
           "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"],
           "weekR": m["week_R_mean"], "wk>=2R": m["weeks_ge_2R"], "pass250": ch["pass_rate"], "fail": ch["fail_rate"],
           "med_pass_d": ch["median_days_to_pass"], "yrs_pos": int((res.daily["end"].resample("YE").last().diff().fillna(
               res.daily["end"].resample("YE").last().iloc[0] - 100_000) > 0).sum())}
    if funded:
        for cap in (0.0, 1.25):
            rf = engine_multi.run(ax, sig, costs, guards(K, orisk, risk_on_initial=True, payout_pct=3.0,
                                                         consistency_pct=35.0, day_profit_cap_pct=cap))
            r, _ = summarize_payouts(rf, "")
            rn = engine_multi.run(ax, sig, costs, guards(K, orisk, risk_on_initial=True, day_profit_cap_pct=cap))
            mf = metrics.summarize(rn, "")
            row.update({f"c{cap}_pay": r["payouts"], f"c{cap}_med_d": r.get("median_days"),
                        f"c{cap}_$": r.get("withdrawn_$"), f"c{cap}_SR": mf["sharpe"], f"c{cap}_wkR": mf["week_R_mean"],
                        f"c{cap}_DD": mf["max_total_dd_pct"]})
    return row, res, m


if __name__ == "__main__":
    ax, xau, idx_legs = load_all()
    SETS = {"XAU only": {},
            "XAU+prior3": {s: PRIOR for s in SYMS[1:]},
            "XAU+prior NAS/DJ": {"NAS100": PRIOR, "DJ30": PRIOR},
            "XAU+selected": {k: v for k, v in SELECTED.items() if v}}
    rows, keep = [], {}
    for name, sets in SETS.items():
        sig = book(xau, idx_legs, sets)
        for K, orisk in ((6, 3.0), (10, 4.0), (12, 5.0)):
            if name == "XAU only" and K != 6:
                continue
            row, res, m = evaluate(ax, sig, K, orisk, name)
            rows.append(row); keep[f"{name}|K{K}"] = (m, res)
            print(row, flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    print(T.T.to_string())
    d = lab.REPORTS / "EXP-086"; d.mkdir(exist_ok=True)
    T.to_csv(d / "summary.csv", index=False)
    lab.save_experiment("EXP-086", {"sets": {k: v for k, v in SETS.items()}}, {k: v[0] for k, v in keep.items()},
                        {k.replace("|", "_").replace("+", "-").replace(" ", "").replace("/", ""): v[1] for k, v in keep.items()})

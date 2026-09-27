"""EXP-087: report of algorithm v2 = XAUUSD best13 + the literature-prior index legs (mon, dip_low20, dip_clv, hi20,
prefomc, tom) on NAS100 and DJ30; one account, shared guards (K6, open risk 3 %), news rule v2.
Challenge mode (compounding) and funded mode (fixed $500, +3 % payouts, 35 % consistency, day cap 1.25 %)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import engine_multi, lab, metrics, symbols  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.payout_sim import chart, summarize_payouts  # noqa: E402
from research.portfolio_v2 import SYMS, book, guards, load_all  # noqa: E402

K, ORISK, CAP = 6, 3.0, 1.25
IDX = {"NAS100": PRIOR, "DJ30": PRIOR}          # GER40 dropped (EXP-085/086: its prior legs are ~0 and add DD)

if __name__ == "__main__":
    ax, xau, idx_legs = load_all()
    sig = book(xau, idx_legs, IDX)
    EXP = "EXP-087"
    if "--pattern" in sys.argv:                                   # v2.1: + walk-forward pattern leg (EXP-090)
        from research.pattern_leg import FILES, pattern_signals
        sig = pd.concat([sig] + [pattern_signals(s_).assign(sym=s_, leg=f"{s_}:pattern") for s_ in FILES]).sort_index(kind="stable")
        EXP = "EXP-096"
    costs = {s: symbols.COSTS[s] for s in SYMS}
    res = engine_multi.run(ax, sig, costs, guards(K, ORISK))
    m = metrics.summarize(res, "v2 challenge mode"); m.update(metrics.monte_carlo_dd(res))
    m["challenge"] = metrics.challenge_sim(res, max_days=250); m["gates"] = metrics.gate_check(m)
    t = res.trades
    eq = res.daily["end"]; ye = eq.resample("YE").last(); ypnl = ye.diff().fillna(ye.iloc[0] - 100_000)
    print({k: m[k] for k in ("trades", "win_rate", "profit_factor", "net_pct", "cagr_pct", "sharpe", "sortino",
                             "max_total_dd_pct", "max_daily_dd_pct", "mc_dd95_pct", "week_R_mean", "weeks_ge_2R")})
    print("challenge:", m["challenge"], "\ngates:", m["gates"])
    print("P&L by year:", {k.year: round(v) for k, v in ypnl.items()})
    by_sym = t.groupby("sym").agg(n=("R", "size"), R=("R", "sum"), pnl=("pnl", "sum")).round(1)
    print("\nby symbol:\n", by_sym.to_string())
    by_leg = t.groupby("leg").agg(n=("R", "size"), avgR=("R", "mean"), R=("R", "sum")).round(3).sort_values("R", ascending=False)
    print("\nby leg:\n", by_leg.to_string())
    # funded mode
    rf = engine_multi.run(ax, sig, costs, guards(K, ORISK, risk_on_initial=True, payout_pct=3.0, consistency_pct=35.0,
                                                 day_profit_cap_pct=CAP))
    r, p = summarize_payouts(rf, f"v2 funded cap {CAP}")
    rn = engine_multi.run(ax, sig, costs, guards(K, ORISK, risk_on_initial=True, day_profit_cap_pct=CAP))
    mf = metrics.summarize(rn, "v2 funded (no payouts)")
    print("\nfunded:", r)
    print("funded perf:", {k: mf[k] for k in ("sharpe", "max_total_dd_pct", "max_daily_dd_pct", "week_R_mean", "weeks_ge_2R")})
    print("payouts per year:", p.groupby(p.time.dt.year).amount.agg(["count", "sum"]).round(0).to_dict())
    d = lab.REPORTS / EXP; d.mkdir(exist_ok=True)
    by_sym.to_csv(d / "by_symbol.csv"); by_leg.to_csv(d / "by_leg.csv"); p.to_csv(d / "payouts.csv", index=False)
    chart(rf, p, d / "payouts.png", f"{EXP} {'v2.1' if EXP != 'EXP-087' else 'v2'} funded 2019–26 (XAU + NAS100/DJ30, cap {CAP} %): {len(p)} payouts, "
          f"median every {p.days.median():.0f} days, ${p.amount.sum():,.0f}")
    mf["payouts"] = r
    lab.save_experiment(EXP, {"book": "v2", "K": K, "orisk": ORISK, "cap": CAP},
                        {"challenge": m, "funded": mf}, {"challenge": res, "funded": rn})

"""QM-LOCKBOX: one-shot evaluation of the two pre-registered candidates (docs/quant_lockbox.md) on 2025-01-01 ..
2026-09-25. Settings are exactly those fixed on DEV; nothing is tuned here."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, engine_multi, lab, legcache, metrics, symbols  # noqa: E402
from research.beta_vs_timing import every_day  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.payout_sim import summarize_payouts  # noqa: E402
from research.portfolio_v2 import book, guards  # noqa: E402
from research.quant import data as Q, portfolio as P, vol as V  # noqa: E402
from research.quant.qm009b_funded_mc import funded  # noqa: E402

L0, L1 = "2025-01-01", "2026-09-25"
OUT = lab.REPORTS / "quant" / "LOCKBOX"
SYMS = ["XAUUSD", "NAS100", "DJ30", "GER40"]


def lockbox_setup():
    m1, legs = legcache.load()
    execs = {"XAUUSD": symbols.prepare("XAUUSD", m1.loc[L0:L1])}
    for s in SYMS[1:]:
        execs[s] = symbols.prepare(s, symbols.load_m1(s).loc[L0:L1])
    ax = engine_multi.align(execs)
    idx_legs = {s: pd.read_pickle(lab.DATA / f"legs_{s}.pkl") for s in SYMS[1:]}
    xau = book_signals(legs, "2019-01-01").assign(sym="XAUUSD")
    sig = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR}).loc[L0:L1]
    return m1, ax, sig


def xau_sigma():
    R = Q.realized("XAUUSD", lockbox=True); R = R[R.rv > 0]
    F = V.walk_forward_rv(R, 2019, 2026)                     # every year forecast with models fitted on earlier years
    s = np.sqrt(F["har_lev_j"])
    ref = s.rolling(250, min_periods=60).median().shift(1)
    return s, ref


def vmult(index, s, ref, shuffle_seed=None):
    m = np.minimum(1.0, (ref / s) ** 2).clip(lower=0.2)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        m = m.groupby(m.index.year, group_keys=False).apply(lambda v: pd.Series(rng.permutation(v.to_numpy()), index=v.index))
    return m.reindex(index.normalize()).fillna(1.0).to_numpy()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m1, ax, sig = lockbox_setup()
    costs = {s: symbols.COSTS[s] for s in SYMS}
    lines = []
    say = lambda *a: (print(*a, flush=True), lines.append(" ".join(str(x) for x in a)))  # noqa: E731

    # ---------------- candidate 1: cushion rule ----------------
    say("=== Candidate 1: cushion-proportional sizing (start 2 %) vs step rule, v2 book, lockbox ===")
    c1 = {}
    for name, kw in (("step", {}), ("cushion2", {"cushion_start_pct": 2.0})):
        res = engine_multi.run(ax, sig, costs, guards(6, 3.0, **kw))
        m = metrics.summarize(res)
        rf = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True, payout_pct=3.0, consistency_pct=35.0,
                                                     day_profit_cap_pct=1.25, **kw))
        r, _ = summarize_payouts(rf, "")
        c1[name] = {"net%": m["net_pct"], "SR": m["sharpe"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
                    "payouts": r["payouts"], "withdrawn": r.get("withdrawn_$"), "breach": r.get("account_breached")}
        say(name, c1[name])
    fixed = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0))
    eq = fixed.daily["end"]; R = (eq.diff().fillna(eq.iloc[0] - 100_000) / 500.0); R = R[R.index.dayofweek < 5].to_numpy()
    say(f"lockbox daily R: mean {R.mean():.3f}, sd {R.std():.3f}, n {len(R)} (DEV was 0.094 / 0.984)")
    mc = {}
    for name, fn in (("step", lambda dd: 1.0 if dd < 6.5 else 0.5), ("cushion2", lambda dd: float(np.clip((8 - dd) / 6, 0.1, 1)))):
        ch = P.policy_mc(R, fn, n_paths=4000)
        pay, lost = funded(R, fn)
        mc[name] = {"P(pass)": ch["p_pass"], "P(fail)": ch["p_fail"], "med_days": ch["median_days"],
                    "funded_payouts/yr": pay, "P(funded lost 2y)": lost}
        say(name, "MC", mc[name])
    ok1 = (abs(c1["cushion2"]["net%"] - c1["step"]["net%"]) <= 0.05 * abs(c1["step"]["net%"]) and
           c1["cushion2"]["SR"] >= 0.95 * c1["step"]["SR"] and c1["cushion2"]["DD%"] <= c1["step"]["DD%"] + 1e-9 and
           abs(c1["cushion2"]["payouts"] - c1["step"]["payouts"]) <= 1 and not c1["cushion2"]["breach"] and
           mc["cushion2"]["P(fail)"] <= mc["step"]["P(fail)"] and mc["cushion2"]["P(funded lost 2y)"] <= mc["step"]["P(funded lost 2y)"])
    say("Candidate 1 verdict:", "PASS" if ok1 else "FAIL")

    # ---------------- candidate 2: gold vol-managed sizing ----------------
    say("\n=== Candidate 2: gold volatility-managed sizing (p = 2), lockbox ===")
    s, ref = xau_sigma()
    x = symbols.prepare("XAUUSD", m1.loc[L0:L1])
    g1 = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390, max_trades_day=8,
                       max_positions=1, max_open_risk_pct=0.5, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0)
    base = every_day(m1, 65, 1360, 1.0, 1.5).loc[L0:L1]
    sr0 = metrics.summarize(engine.run(x, base, g1))["sharpe"]
    sr2 = metrics.summarize(engine.run(x, base.assign(risk_mult=vmult(base.index, s, ref)), g1))["sharpe"]
    ph = [metrics.summarize(engine.run(x, base.assign(risk_mult=vmult(base.index, s, ref, k)), g1))["sharpe"] for k in range(20)]
    p = float(np.mean(np.array(ph) >= sr2))
    say(f"XAU always-long: SR unmanaged {sr0}, managed {sr2}, placebo mean {np.mean(ph):.3f}, p {p:.2f}")
    v2 = {}
    for name, managed in (("v2", False), ("v2 + gold vol sizing", True)):
        sg = sig.copy()
        if managed:
            k = (sg.sym == "XAUUSD").to_numpy()
            rm = sg.get("risk_mult", pd.Series(1.0, index=sg.index)).fillna(1.0).to_numpy().copy()
            rm[k] = rm[k] * vmult(sg.index[k], s, ref)
            sg["risk_mult"] = rm
        m = metrics.summarize(engine_multi.run(ax, sg, costs, guards(6, 3.0)))
        v2[name] = {"SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"]}
        say(name, v2[name])
    ok2 = sr2 > sr0 and p <= 0.10 and v2["v2 + gold vol sizing"]["SR"] >= v2["v2"]["SR"] and \
        v2["v2 + gold vol sizing"]["DD%"] <= v2["v2"]["DD%"] + 1e-9
    say("Candidate 2 verdict:", "PASS" if ok2 else "FAIL")
    (OUT / "result.txt").write_text("\n".join(lines), encoding="utf-8")

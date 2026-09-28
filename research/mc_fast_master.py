"""EXP-107: "fast to Master" Monte Carlo — FundingPips 2 Step Standard $10k (P1 +8 %, P2 +5 %, daily loss 5 % of
max(day-open balance, equity) incl. floating, static max loss 10 %, >= 3 trading days per phase, no time limit).
User request: risk 3 % per trade, no internal drawdown limits (only FundingPips' own limits remain).

Trade stream = algorithm v2.2 book (XAU 12 legs + NAS100/DJ30 6 legs, news v2, weekend flat) run by the multi-symbol
engine on a $10k account with fixed-$ risk = r % of $10k, the SAME trades at every r (open-risk cap 6 r, no daily
soft/hard stops, no total stop, no cushion) -> per day: P&L, intraday worst equity, FP day reference.
Challenge replay: a phase fails the moment intraday equity touches ref x (1 - 5 %) or $9,000; passes when closed
equity >= target with >= 3 trading days. Two resamplings: (1) every historical start day 2019-2026, (2) stationary
block bootstrap of days (mean block 10), 20,000 paths, max 500 trading days per attempt.
Variants: r = 0.5 / 1 / 2 / 3 %, plus r = 3 % with at most 2 open positions."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine_multi, lab, symbols  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.portfolio_v2 import SYMS, book, load_all  # noqa: E402
from research.quant.validate import stationary_bootstrap  # noqa: E402

ACC = 10_000.0
OUT = lab.REPORTS / "EXP-107"
rng = np.random.default_rng(107)


def day_table(ax, sig, r, K=6):
    gd = dataclasses.replace(g(True, K), initial_balance=ACC, risk_pct=r, risk_on_initial=True, max_open_risk_pct=6 * r,
                             daily_soft_pct=1e6, daily_hard_pct=1e6, total_stop_pct=1e6, total_derisk_pct=1e6,
                             cushion_start_pct=0.0)
    res = engine_multi.run(ax, sig, {s: symbols.COSTS[s] for s in SYMS}, gd)
    d = res.daily[res.daily.index.dayofweek < 5]
    prev = d["end"].shift(1).fillna(ACC)
    traded = pd.Series(1, index=res.trades.entry_time.dt.normalize()).groupby(level=0).size().reindex(d.index).fillna(0) > 0
    return pd.DataFrame({"pnl": d["end"] - prev, "worst": d["min"] - prev, "ref_off": d["start"] - prev,
                         "traded": traded.astype(int)}), res


def attempt(pnl, worst, ref_off, traded, idx):
    """one attempt through phase 1 and 2 along the day sequence idx -> (outcome, days used)"""
    k, used = 0, 0
    for target in (0.08, 0.05):
        eq, days = ACC, 0
        while True:
            if k >= len(idx):
                return "timeout", used
            j = idx[k]; k += 1; used += 1
            ref = eq + ref_off[j]
            if eq + worst[j] <= ref * 0.95 or eq + worst[j] <= ACC * 0.90:
                return "fail", used
            eq += pnl[j]; days += traded[j]
            if eq - ACC >= target * ACC and days >= 3:
                break
    return "pass", used


def summarize(out):
    o = pd.Series([a for a, _ in out]); d = np.array([u for a, u in out if a == "pass"])
    p = (o == "pass").mean()
    return {"P(pass both)": round(p, 3), "P(fail)": round((o == "fail").mean(), 3), "P(timeout 500d)": round((o == "timeout").mean(), 3),
            "median_days_to_Master": int(np.median(d)) if len(d) else None, "p90_days": int(np.quantile(d, 0.9)) if len(d) else None,
            "expected_attempts": round(1 / p, 2) if p > 0 else None}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ax, xau, idx_legs = load_all()
    sig = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR})
    rows = []
    for label, r, K in (("0.5 %", 0.5, 6), ("1 %", 1.0, 6), ("2 %", 2.0, 6), ("3 %", 3.0, 6), ("3 %, max 2 open", 3.0, 2)):
        D, res = day_table(ax, sig, r, K)
        a = [D[c].to_numpy() for c in ("pnl", "worst", "ref_off", "traded")]
        n = len(D)
        hist = [attempt(*a, np.arange(s, n)) for s in range(0, n - 60)]
        boot = []
        for _ in range(20_000):
            ix = stationary_bootstrap(n, mean_block=10, rng=rng)
            while len(ix) < 500:
                ix = np.r_[ix, stationary_bootstrap(n, mean_block=10, rng=rng)]
            boot.append(attempt(*a, ix[:500]))
        worst_day = (D.worst / (ACC + D.pnl.cumsum().shift(1).fillna(0))).min() * 100
        row = {"risk/trade": label, "trades": len(res.trades), "avg R": round(res.trades.R.mean(), 3),
               "worst intraday day %": round(worst_day, 1), "days with <= -5 %": int((D.worst <= -0.05 * ACC).sum()),
               **{f"hist {k}": v for k, v in summarize(hist).items()}, **{f"MC {k}": v for k, v in summarize(boot).items()}}
        rows.append(row); print(row, flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "fast_master.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.T.to_string())

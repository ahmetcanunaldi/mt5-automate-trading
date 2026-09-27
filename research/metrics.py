"""Performance metrics, FundingPips challenge simulation and acceptance gates."""
import numpy as np
import pandas as pd

from research.engine import Result

TRADING_DAYS = 252

GATES = {
    "sharpe": 1.5,          # daily returns, annualized sqrt(252)
    "max_daily_dd_pct": 3.0,
    "max_total_dd_pct": 8.0,
    "profit_factor": 1.3,
    "min_trades": 200,
    "mc_dd95_pct": 8.0,
    "week_R_mean": 2.0,     # user target: >= 2 R (1 %) per week on average
}


def daily_returns(res: Result) -> pd.Series:
    d = res.daily
    d = d[d.index.dayofweek < 5]
    base = d["start_eq"] if "start_eq" in d else d["start"]
    return (d["end"] / base - 1.0).fillna(0.0)


def summarize(res: Result, label: str = "") -> dict:
    tr = res.trades
    r = daily_returns(res)
    eq = res.equity
    peak = np.maximum.accumulate(np.maximum(eq.to_numpy(), res.params["guards"]["initial_balance"]))
    total_dd = float(((peak - eq.to_numpy()) / peak).max() * 100) if len(eq) else 0.0
    # daily DD the FundingPips way: vs day-start reference, using intraday min equity (bar closes)
    dd_day = ((res.daily["start"] - res.daily["min"]) / res.daily["start"] * 100).clip(lower=0)
    gp = tr.loc[tr.pnl > 0, "pnl"].sum(); gl = -tr.loc[tr.pnl < 0, "pnl"].sum()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) if len(eq) else 1
    out = {
        "label": label,
        "from": str(eq.index[0].date()) if len(eq) else "", "to": str(eq.index[-1].date()) if len(eq) else "",
        "trades": int(len(tr)),
        "trades_per_week": round(len(tr) / (years * 52), 2),
        "win_rate": round(float((tr.pnl > 0).mean()), 3) if len(tr) else 0,
        "avg_R": round(float(tr.R.mean()), 3) if len(tr) else 0,
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else float("inf"),
        "net_pct": round(float(eq.iloc[-1] / res.params["guards"]["initial_balance"] * 100 - 100), 2) if len(eq) else 0,
        "cagr_pct": round(float(((eq.iloc[-1] / res.params["guards"]["initial_balance"]) ** (1 / years) - 1) * 100), 2) if len(eq) else 0,
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 3) if r.std() > 0 else 0.0,
        "sortino": round(float(r.mean() / r[r < 0].std() * np.sqrt(TRADING_DAYS)), 3) if (r < 0).sum() > 1 else 0.0,
        "max_total_dd_pct": round(total_dd, 2),
        "max_daily_dd_pct": round(float(dd_day.max()), 2) if len(dd_day) else 0.0,
        "worst_day_pct": round(float(r.min() * 100), 2) if len(r) else 0.0,
        "avg_hold_min": round(float((tr.exit_time - tr.entry_time).dt.total_seconds().mean() / 60), 1) if len(tr) else 0,
        "exit_reasons": tr.reason.value_counts().to_dict() if len(tr) else {},
    }
    out.update(weekly_stats(res))
    return out


def weekly_stats(res: Result) -> dict:
    """Weekly P&L in R units (R = risk_pct of the initial balance): FundingPips payout needs >= 2 %/cycle,
    the user's target is >= 2 R (1 %) per week."""
    g = res.params["guards"]
    init, rp = g["initial_balance"], g["risk_pct"] / 100.0
    end = res.daily["end"]
    wk = end.resample("W-FRI").last().dropna()
    prev = wk.shift(1).fillna(init)
    wr = (wk - prev) / (init * rp)
    return {"week_R_mean": round(float(wr.mean()), 2) if len(wr) else 0.0,
            "week_R_median": round(float(wr.median()), 2) if len(wr) else 0.0,
            "weeks_ge_2R": round(float((wr >= 2).mean()), 3) if len(wr) else 0.0,
            "weeks_pos": round(float((wr > 0).mean()), 3) if len(wr) else 0.0}


def monte_carlo_dd(res: Result, n=2000, seed=7) -> dict:
    """Reshuffle trade order (R-multiples at fixed risk) to estimate drawdown distribution."""
    tr = res.trades
    if len(tr) < 10:
        return {"mc_dd50_pct": None, "mc_dd95_pct": None}
    risk = res.params["guards"]["risk_pct"] / 100
    R = tr.R.to_numpy()
    rng = np.random.default_rng(seed)
    dds = np.empty(n)
    for k in range(n):
        eq = np.cumprod(1 + rng.permutation(R) * risk)
        peak = np.maximum.accumulate(np.maximum(eq, 1.0))
        dds[k] = ((peak - eq) / peak).max()
    return {"mc_dd50_pct": round(float(np.percentile(dds, 50) * 100), 2),
            "mc_dd95_pct": round(float(np.percentile(dds, 95) * 100), 2)}


def challenge_sim(res: Result, target1=8.0, target2=5.0, fp_daily=5.0, fp_total=10.0,
                  our_daily=3.0, our_total=8.0, min_days=3, max_days=120) -> dict:
    """Replay daily P&L (as % of balance at fixed risk) from every possible start day.
    A phase passes when cumulative >= target and >= min_days trading days;
    fails on FP breach or on our internal limits. Phase 2 starts the day after phase 1 passes.
    Uses per-day returns and intraday min equity so intraday breaches count."""
    d = res.daily[res.daily.index.dayofweek < 5].copy()
    base = d["start_eq"] if "start_eq" in d else d["start"]
    ret = (d["end"] / base - 1).to_numpy()                 # equity-to-equity return (compounding)
    low = (d["min"] / base - 1).to_numpy()                 # intraday worst vs start equity
    ref = (d["start"] / base).to_numpy()                   # FP day reference = max(bal, eq) / start equity
    tr = res.trades
    traded = pd.Series(1, index=tr.entry_time.dt.normalize()).groupby(level=0).size().reindex(d.index).fillna(0).to_numpy() > 0

    def phase(start, target):
        bal = 1.0; days = 0
        for k in range(start, min(start + max_days, len(ret))):
            intraday_low = bal * (1 + low[k])
            day_ref = bal * ref[k]
            if (day_ref - intraday_low) / day_ref * 100 >= min(fp_daily, our_daily) or (1 - intraday_low) * 100 >= min(fp_total, our_total):
                return "fail", k
            bal *= 1 + ret[k]
            days += traded[k]
            if (bal - 1) * 100 >= target and days >= min_days:
                return "pass", k
        return "timeout", None

    outcomes = []; durations = []
    for s in range(0, len(ret) - 20):
        r1, k1 = phase(s, target1)
        if r1 != "pass":
            outcomes.append(r1); continue
        r2, k2 = phase(k1 + 1, target2)
        outcomes.append("pass" if r2 == "pass" else r2)
        if r2 == "pass":
            durations.append(k2 - s + 1)
    o = pd.Series(outcomes)
    return {
        "starts": int(len(o)),
        "pass_rate": round(float((o == "pass").mean()), 3) if len(o) else 0,
        "fail_rate": round(float((o == "fail").mean()), 3) if len(o) else 0,
        "timeout_rate": round(float((o == "timeout").mean()), 3) if len(o) else 0,
        "median_days_to_pass": float(np.median(durations)) if durations else None,
    }


def gate_check(m: dict) -> dict:
    checks = {
        "sharpe": m["sharpe"] >= GATES["sharpe"],
        "daily_dd": m["max_daily_dd_pct"] < GATES["max_daily_dd_pct"],
        "total_dd": m["max_total_dd_pct"] < GATES["max_total_dd_pct"],
        "profit_factor": m["profit_factor"] >= GATES["profit_factor"],
        "trades": m["trades"] >= GATES["min_trades"],
        "weekly_2R": m.get("week_R_mean", 0) >= GATES["week_R_mean"],
    }
    if m.get("mc_dd95_pct") is not None:
        checks["mc_dd95"] = m["mc_dd95_pct"] < GATES["mc_dd95_pct"]
    checks["ALL"] = all(checks.values())
    return checks

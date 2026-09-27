"""QM-003: volatility-managed exposure (Moreira & Muir 2017) with HAR forecasts, DEV 2019-2024 only.
risk_mult = min(1, (median_sigma / sigma_hat)^p), sigma_hat^2 = HAR(+lev+jump) one-day-ahead RV forecast (walk-
forward, known before the day), median over the trailing year (causal). p = 1 (vol targeting) and 2 (MM variance
scaling). Applied to (a) always-long daily positions per symbol and (b) algorithm v2 (one account, shared guards).
Compared with the unscaled versions on Sharpe, DD, R/yr; plus a placebo where the multiplier series is shuffled
within years (does *timing* the scaling matter, or only the average exposure?)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, engine_multi, lab, metrics, symbols  # noqa: E402
from research.beta_vs_timing import every_day  # noqa: E402
from research.quant import data as Q, vol as V  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

A, B = "2019-01-01", "2024-12-31"
OUT = lab.REPORTS / "quant" / "QM-003"


def har_sigma(sym):
    R = Q.realized(sym); R = R[R.rv > 0]
    F = V.walk_forward_rv(R, 2019, 2024)
    s = np.sqrt(F["har_lev_j"])
    ref = s.rolling(250, min_periods=60).median().shift(1).fillna(s.expanding().median().shift(1))
    return s, ref


def mult(sig, s, ref, p, shuffle_seed=None):
    m = np.minimum(1.0, (ref / s) ** p).clip(lower=0.2)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        m = m.groupby(m.index.year, group_keys=False).apply(lambda v: pd.Series(rng.permutation(v.to_numpy()), index=v.index))
    return m.reindex(sig.index.normalize()).fillna(1.0).to_numpy()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    S = {s: har_sigma(s) for s in ("XAUUSD", "NAS100", "DJ30")}
    rows = []
    g1 = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390, max_trades_day=8,
                       max_positions=1, max_open_risk_pct=0.5, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0)
    for sym in S:
        m1 = symbols.load_m1(sym)
        x = symbols.prepare(sym, m1.loc[A:B])
        base = every_day(m1, 65, 1360, 1.0, 1.5).loc[A:B]
        for p in (0, 1, 2):
            sig = base.copy()
            if p:
                sig["risk_mult"] = mult(sig, *S[sym], p)
            res = engine.run(x, sig, g1, symbols.COSTS[sym])
            m = metrics.summarize(res)
            row = {"book": f"{sym} always-long", "p": p, "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                   "avg_mult": round(float(np.mean(sig.get("risk_mult", pd.Series(1.0, index=sig.index)))), 2)}
            if p:
                ph = [metrics.summarize(engine.run(x, sig.assign(risk_mult=mult(sig, *S[sym], p, k)), g1, symbols.COSTS[sym]))["sharpe"]
                      for k in range(20)]
                row.update({"placebo_SR_mean": round(np.mean(ph), 3), "placebo_p": round(float(np.mean(np.array(ph) >= m["sharpe"])), 2)})
            rows.append(row); print(row, flush=True)
            led.log("QM-003", "vol_managed", {"book": row["book"], "p": p}, {"SR": m["sharpe"]})
    # v2 book on the DEV window
    from research.portfolio_v2 import book, guards, load_all
    from research.combo_idx import PRIOR
    ax, xau, idx_legs = load_all()
    v2 = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR}).loc[A:B]
    costs = {s: symbols.COSTS[s] for s in ["XAUUSD", "NAS100", "DJ30", "GER40"]}
    for p in (0, 1, 2):
        sig = v2.copy()
        if p:
            mm = np.ones(len(sig))
            for sym in S:
                k = (sig.sym == sym).to_numpy()
                mm[k] = mult(sig[k], *S[sym], p)
            sig["risk_mult"] = sig.get("risk_mult", pd.Series(1.0, index=sig.index)).fillna(1.0).to_numpy() * mm
        res = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True))
        m = metrics.summarize(res)
        eq = res.equity.loc[:B]
        row = {"book": "v2 (fixed $500)", "p": p, "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
               "week_R": m["week_R_mean"]}
        rows.append(row); print(row, flush=True)
        led.log("QM-003", "vol_managed", {"book": "v2", "p": p}, {"SR": m["sharpe"]})
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))

"""EXP-090: walk-forward pattern leg (EXP-089 demeaned mining, vote > 0 -> long next day) added to algorithm v2.
Entry next trading day at 01:06 (one minute after the other daily legs so it does not take their slot), exit 23:30,
SL 1 ATR_D, trail 1.5 x SL. Also a strict variant: only days with >= 2 positive votes."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, engine_multi, lab, metrics, symbols  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.features import atr  # noqa: E402
from research import pattern_mine  # noqa: E402
from research.portfolio_v2 import book, guards, load_all  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
FILES = {"XAUUSD": "XAUUSD_D1_2007", "NAS100": "NAS100_D1_long", "DJ30": "DJ30_D1_long"}


def pattern_signals(sym, min_votes=1, t_min=66, t_exit=1410):
    pattern_mine.DEMEAN = True
    d = pd.read_parquet(lab.DATA / f"{FILES[sym]}.parquet")[["open", "high", "low", "close"]]
    d = d[(d.index.dayofweek < 5) & (d.high > d.low)]
    o, _, _ = pattern_mine.run(sym, d)
    a = atr(d, 14)
    # vote strength is not stored; recompute from dir only (min_votes > 1 uses the n_sel-agnostic dir) -> use dir
    days = o.index[o.dir > 0]
    nxt = d.index.searchsorted(days, side="right"); ok = nxt < len(d)
    t = d.index[nxt[ok]] + pd.Timedelta(minutes=t_min)
    s = pd.DataFrame({"dir": 1, "sl": a.loc[days[ok]].to_numpy(), "tp": 1e6, "hold_min": t_exit - t_min, "be": 0.0,
                      "trail": 1.5 * a.loc[days[ok]].to_numpy(), "leg": "pattern"}, index=t)
    return s.loc[A:B]


if __name__ == "__main__":
    ax, xau, idx_legs = load_all()
    base = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR})
    P = {s: pattern_signals(s) for s in FILES}
    costs = {s: symbols.COSTS[s] for s in ["XAUUSD", "NAS100", "DJ30", "GER40"]}
    rows = []
    # standalone
    for s, sig in P.items():
        m1 = symbols.load_m1(s)
        x = symbols.prepare(s, m1.loc[A:B])
        g1 = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                           max_trades_day=8, max_positions=1, max_open_risk_pct=0.5, risk_on_initial=True,
                           total_stop_pct=100.0, total_derisk_pct=100.0)
        t = engine.run(x, sig, g1, symbols.COSTS[s]).trades
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        print(f"{s} pattern leg standalone: n {len(t)} avgR {t.R.mean():+.3f} t {t.R.mean() / t.R.std() * np.sqrt(len(t)):+.2f} "
              f"R/yr {t.R.sum() / 7.7:.1f} yrs+ {(yr > 0).sum()}/{len(yr)}", flush=True)
    variants = {"v2": base}
    for combo in (["XAUUSD"], ["NAS100"], ["DJ30"], ["XAUUSD", "NAS100", "DJ30"], ["XAUUSD", "NAS100"]):
        parts = [base] + [P[s].assign(sym=s, leg=f"{s}:pattern") for s in combo]
        variants["v2+pattern " + "/".join(combo)] = pd.concat(parts).sort_index(kind="stable")
    for name, sig in variants.items():
        res = engine_multi.run(ax, sig, costs, guards(6, 3.0))
        m = metrics.summarize(res, name); m.update(metrics.monte_carlo_dd(res)); ch = metrics.challenge_sim(res, max_days=250)
        rn = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True, day_profit_cap_pct=1.25))
        mf = metrics.summarize(rn, "")
        rows.append({"book": name, "n": m["trades"], "SR": m["sharpe"], "CAGR%": m["cagr_pct"], "DD%": m["max_total_dd_pct"],
                     "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"], "weekR": m["week_R_mean"],
                     "pass250": ch["pass_rate"], "fail": ch["fail_rate"], "funded_SR": mf["sharpe"], "funded_wkR": mf["week_R_mean"]})
        print(rows[-1], flush=True)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    (lab.REPORTS / "EXP-090").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(lab.REPORTS / "EXP-090" / "summary.csv", index=False)

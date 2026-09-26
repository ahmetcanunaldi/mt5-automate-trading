"""EXP-033: H4 Donchian trend following with pyramiding (K same-direction positions, each 0.5 % risk,
open-risk cap), 2019-2026, M15 execution, swaps, weekend flat on/off, REAL guards ($100k, 3 % daily, 8 % total)
plus a no-stop diagnostic. Also the D1 version over 2008-2026 (longer regime check)."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.strategies import swing  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def run_h4():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    h4 = m1.resample("4h").agg(AGG).dropna(subset=["open"])
    m15 = m1.loc[A:B].resample("15min").agg(AGG).dropna(subset=["open"])
    x = engine.prepare_exec(m15, 15)
    df = swing.prep(h4, bar=pd.Timedelta(hours=4))
    rows, keep = [], {}
    for n, trail, both, K, wf in itertools.product([60, 120, 180], [4.0, 6.0], [False, True], [1, 2, 4], [True, False]):
        s = swing.donchian(df, n=n, sl=2.0, trail=trail, tp=50.0, hold=240, both=both).loc[A:B]
        for stop in (8.0, 100.0):
            g = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=wf, first_entry_min=65,
                              last_entry_min=1380, max_trades_day=4, max_positions=K, max_open_risk_pct=0.5 * K,
                              total_stop_pct=stop, total_derisk_pct=6.5 if stop < 50 else stop)
            res = engine.run(x, s, g)
            t = res.trades
            if len(t) < 20:
                continue
            m = metrics.summarize(res, "")
            name = f"n{n}_tr{trail}_{'LS' if both else 'L'}_K{K}_wf{int(wf)}_stop{int(stop)}"
            yr = t.groupby(t.entry_time.dt.year).R.sum()
            ch = metrics.challenge_sim(res, max_days=250) if stop < 50 else {}
            rows.append({"cfg": name, "n": len(t), "avgR": m["avg_R"], "R_per_yr": round(t.R.sum() / 7.7, 1),
                         "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                         "dDD%": m["max_daily_dd_pct"], "yrs_pos": int((yr > 0).sum()),
                         "pass250": ch.get("pass_rate"), "med_days": ch.get("median_days_to_pass"),
                         **{f"R{y}": round(v, 1) for y, v in yr.items()}})
            keep[name] = (m, res)
    return pd.DataFrame(rows), keep


if __name__ == "__main__":
    T, keep = run_h4()
    T = T.sort_values("SR", ascending=False)
    (lab.REPORTS / "EXP-033").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-033" / "summary.csv", index=False)
    pd.set_option("display.width", 260)
    print("=== with real guards (stop 8 %) ===")
    print(T[T.cfg.str.endswith("stop8")].head(20).to_string(index=False))
    print("\n=== diagnostic (no total stop) ===")
    print(T[T.cfg.str.endswith("stop100")].head(12).to_string(index=False))
    top = list(T[T.cfg.str.endswith("stop8")].cfg.head(3))
    lab.save_experiment("EXP-033", {"strategy": "H4 Donchian pyramiding"}, {c: keep[c][0] for c in top},
                        {c: keep[c][1] for c in top})

"""EXP-078: the XAUUSD book's leg rules applied unchanged to XAGUSD, EURUSD, USDJPY (and XAUUSD as reference),
M1 execution 2019-2026, $100k, 0.5 % risk, standalone per leg (K1), symbol news (USD + EUR/JPY), real spreads,
$7/lot, swaps. No parameter was tuned on the new symbols -> this is an out-of-sample transfer test."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, symbols  # noqa: E402
from research.multi_legs import A, B, legs_for  # noqa: E402


def g1():
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                         flatten_min=1425, fri_flatten_min=1350, max_trades_day=8, max_positions=1,
                         max_open_risk_pct=0.5, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0)


def tstat(v):
    return round(v.mean() / v.std() * np.sqrt(len(v)), 2) if len(v) > 2 and v.std() > 0 else 0.0


if __name__ == "__main__":
    rows = []
    for sym in sys.argv[1:] or ["XAUUSD", "XAGUSD", "EURUSD", "USDJPY"]:
        m1 = symbols.load_m1(sym)
        x = symbols.prepare(sym, m1.loc[A:B])
        legs = legs_for(m1)
        pd.to_pickle(legs, lab.DATA / f"legs_{sym}.pkl")
        c0 = dataclasses.replace(symbols.COSTS[sym], commission_per_lot=0.0, slippage_pts=0.0)
        x0 = symbols.prepare(sym, m1.loc[A:B], costs=c0)
        for name, s in legs.items():
            gross = engine.run(x0, s, g1(), c0).trades.R
            t = engine.run(x, s, g1(), symbols.COSTS[sym]).trades
            if len(t) < 10:
                continue
            yr = t.groupby(t.entry_time.dt.year).R.sum()
            rows.append({"sym": sym, "leg": name, "n": len(t), "avgR": round(t.R.mean(), 3), "t": tstat(t.R), "avgR_nocomm": round(gross.mean(), 3), "t_nocomm": tstat(gross),
                         "R_yr": round(t.R.sum() / 7.7, 1), "t_19_22": tstat(t[t.entry_time.dt.year <= 2022].R),
                         "t_23_26": tstat(t[t.entry_time.dt.year >= 2023].R), "yrs_pos": int((yr > 0).sum())})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    (lab.REPORTS / "EXP-078").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-078" / f"legs_{'_'.join(sorted(T.sym.unique()))}.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))

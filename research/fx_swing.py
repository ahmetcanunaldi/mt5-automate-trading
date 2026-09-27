"""EXP-080: multi-day (swing) families on XAGUSD / EURUSD / USDJPY (+ XAUUSD reference), where costs are small
relative to the move. D1 decisions (bars built from M1), M1 execution 2019-2026, weekend flat (state strategies
re-enter on Monday), swaps charged, symbol news rules, $100k, 0.5 % risk, standalone K1.
Pre-specified grid (no per-symbol tuning): Donchian 20/55/100 (SL 2 ATR, trail 3 ATR), EMA20/100 state,
TSMOM 60/120 weekly, weekly-high/low break, RSI2 and Bollinger mean reversion (EMA200 filter)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, symbols  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.multi_scan import tstat  # noqa: E402
from research.strategies import swing  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
FAM = {
    "don20": lambda d: swing.donchian(d, 20, 2.0, 3.0, 20.0, 40),
    "don55": lambda d: swing.donchian(d, 55, 2.0, 3.0, 20.0, 60),
    "don100": lambda d: swing.donchian(d, 100, 2.0, 3.0, 20.0, 80),
    "ema20_100": lambda d: swing.ema_state(d, 20, 100, 2.0, 3.0, 20.0, 20),
    "tsmom60": lambda d: swing.tsmom(d, "ret60", 2.0, 0.0, 20.0, 5),
    "tsmom120": lambda d: swing.tsmom(d, "ret120", 2.0, 0.0, 20.0, 5),
    "week_break": lambda d: swing.week_break(d),
    "rsi2": lambda d: swing.rsi2(d),
    "bb_mr": lambda d: swing.bollinger_mr(d),
}


def gsw():
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                         max_trades_day=8, max_positions=1, max_open_risk_pct=0.5, risk_on_initial=True,
                         total_stop_pct=100.0, total_derisk_pct=100.0)


if __name__ == "__main__":
    rows = []
    for sym in ("XAUUSD", "XAGUSD", "EURUSD", "USDJPY"):
        m1 = symbols.load_m1(sym)
        x = symbols.prepare(sym, m1.loc[A:B])
        d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
        df = swing.prep(d1, bar=pd.Timedelta(days=1))
        sigs = {}
        for fam, fn in FAM.items():
            s = fn(df).loc[A:B]
            # enter at 01:05 server of the next day (the D1 signal index is the next day's 00:00)
            wk = s.index.dayofweek >= 5                      # Friday decisions -> Monday 01:05
            s.index = s.index + pd.to_timedelta(np.where(wk, 7 - s.index.dayofweek, 0), unit="D") + pd.Timedelta(minutes=65)
            for d, tag in ((1, "L"), (-1, "S")):
                sd = s[s.dir == d].copy(); sd["leg"] = f"{fam}_{tag}"
                if len(sd) >= 20:
                    sigs[f"{fam}_{tag}"] = sd
        pd.to_pickle(sigs, lab.DATA / f"swing_{sym}.pkl")
        for name, s in sigs.items():
            t = engine.run(x, s, gsw(), symbols.COSTS[sym]).trades
            if len(t) < 15:
                continue
            yr = t.groupby(t.entry_time.dt.year).R.sum()
            rows.append({"sym": sym, "leg": name, "n": len(t), "avgR": round(t.R.mean(), 3), "t": tstat(t.R),
                         "R_yr": round(t.R.sum() / 7.7, 1), "t_19_22": tstat(t[t.entry_time.dt.year <= 2022].R),
                         "t_23_26": tstat(t[t.entry_time.dt.year >= 2023].R), "yrs_pos": int((yr > 0).sum())})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    (lab.REPORTS / "EXP-080").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-080" / "swing_legs.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.sort_values(["sym", "t"], ascending=[True, False]).to_string(index=False))

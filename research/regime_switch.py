"""EXP-051: bear-market protection for the long-biased legs, daily bars 2008-2026 (includes the 2013-18 bear).
(a) Donchian trend (daily proxy of the H4 leg): long-only vs regime-switched (long above EMA200, short below)
(b) Friday-long / turn-of-month with and without a 'skip when close < EMA200' filter.
Swing engine on D1 bars, swaps, $100k, no total stop (diagnostic)."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.run_swing import COSTS, exec_d1, guards  # noqa: E402
from research.strategies import swing  # noqa: E402

ERAS = [(2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026)]


def report(name, res):
    t = res.trades
    m = metrics.summarize(res, name)
    row = {"cfg": name, "n": len(t), "avgR": m["avg_R"], "SR": m["sharpe"], "DD%": m["max_total_dd_pct"],
           "yrs_pos": int((t.groupby(t.entry_time.dt.year).R.sum() > 0).sum())}
    for a, b in ERAS:
        e = t[(t.entry_time.dt.year >= a) & (t.entry_time.dt.year <= b)]
        row[f"R/yr {a}-{str(b)[2:]}"] = round(e.R.sum() / (b - a + 1), 1)
    return row


if __name__ == "__main__":
    d, x = exec_d1()
    df = swing.prep(pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet"))
    bull = df.close > df.ema200
    rows = []
    for n, trail in itertools.product([30, 55], [4.0, 6.0]):
        base = swing.donchian(df, n=n, sl=2.0, trail=trail, tp=50.0, hold=60, both=True)
        dec = base.index - pd.Timedelta(days=1)
        reg = bull.reindex(dec).to_numpy()
        long_only = base[base.dir == 1]
        switched = base[((base.dir == 1) & reg) | ((base.dir == -1) & ~reg)]
        for nm, s in (("long_only", long_only), ("regime_LS", switched), ("both_always", base)):
            rows.append(report(f"donchian{n}_tr{trail}_{nm}", engine.run(x, s.loc["2008-02":], guards(True), COSTS)))
    fri = swing.weekday(df, dow=4, direction=1, sl=1.5)
    tom = swing.turn_of_month(df, days_before=0, days_after=3, sl=2.0)
    for nm, s in (("friday", fri), ("tom", tom)):
        reg = bull.reindex(s.index - pd.Timedelta(days=1)).to_numpy()
        rows.append(report(f"{nm}_always", engine.run(x, s.loc["2008-02":], guards(True), COSTS)))
        rows.append(report(f"{nm}_bull_only", engine.run(x, s[reg].loc["2008-02":], guards(True), COSTS)))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    (lab.REPORTS / "EXP-051").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-051" / "summary.csv", index=False)

"""EXP-039: H1 swing/intraday battery 2019-2026 (signals on H1 bars from the M1 export, execution on M15 bars),
$100k, swing engine with swaps, weekend-flat vs not, no total stop (diagnostic). 'hold' is in H4 bars."""
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.strategies import swing  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
COSTS = engine.Costs()

GRIDS = {
    "donchian": dict(n=[24, 72, 168], sl=[2.0, 3.0], trail=[3.0, 5.0], hold=[12, 120], filt=[None, "ema200"],
                     both=[True, False]),
    "ema_state": dict(fast=[10, 20, 50], slow=[50, 100, 200], sl=[2.0, 3.0], trail=[0.0, 4.0], hold=[6, 48],
                      both=[True, False]),
    "rsi2": dict(lo=[5, 10], hi=[90, 95], sl=[2.0, 3.0], tp=[1.0, 2.0], hold=[6, 12], both=[True, False]),
    "bollinger_mr": dict(sl=[2.0, 3.0], tp=[1.0, 2.0], hold=[6, 12], both=[True, False]),
}

if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    h4 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    m15 = m1.loc[A:B].resample("15min").agg(AGG).dropna(subset=["open"])
    x = engine.prepare_exec(m15, 15, costs=COSTS)
    df = swing.prep(h4, bar=pd.Timedelta(hours=1))
    rows = []
    for wf in (True, False):
        g = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=wf, first_entry_min=65,
                          last_entry_min=1380, max_trades_day=3, total_stop_pct=100.0, total_derisk_pct=100.0)
        for fam, grid in GRIDS.items():
            keys = list(grid)
            for vals in itertools.product(*[grid[k] for k in keys]):
                p = dict(zip(keys, vals))
                if fam == "ema_state" and p["fast"] >= p["slow"]:
                    continue
                s = swing.LIB[fam](df, **p).loc[A:B]
                res = engine.run(x, s, g, COSTS)
                t = res.trades
                if len(t) < 30:
                    continue
                m = metrics.summarize(res, "")
                yr = t.groupby(t.entry_time.dt.year).R.mean()
                rows.append({"wflat": wf, "family": fam, "cfg": json.dumps(p), "n": len(t), "avgR": m["avg_R"],
                             "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "SR": m["sharpe"],
                             "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "yrs_pos": int((yr > 0).sum()),
                             "n_yrs": len(yr), **{f"R{y}": round(v, 3) for y, v in yr.items()}})
            print(wf, fam, "done", flush=True)
    T = pd.DataFrame(rows).sort_values(["yrs_pos", "t"], ascending=False)
    (lab.REPORTS / "EXP-039").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-039" / "summary.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 80)
    print(f"configs {len(T)}; >=7/8 years positive: {(T.yrs_pos >= 7).sum()}; t>3: {(T.t > 3).sum()}")
    print(T.head(25).to_string(index=False))
    print(T.groupby(["wflat", "family"]).t.describe()[["count", "mean", "max"]].round(2))

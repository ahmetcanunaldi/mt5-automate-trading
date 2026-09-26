"""EXP-036b: Friday-close drift: long Friday 23:05 server -> exit 23:55 (before the weekend close), SL k x ATR_M15.
Also the same window on Mon-Thu as a control. M1 2019-2026, $100k, real bar spreads."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"

if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    a15 = atr(m15, 14)
    ex = m1.loc[A:B]
    x = engine.prepare_exec(ex, 1)
    g = engine.Guards(initial_balance=100_000, last_entry_min=1390, flatten_min=1437, fri_flatten_min=1500,
                      max_trades_day=3, total_stop_pct=100.0, total_derisk_pct=100.0)
    days = pd.DatetimeIndex(np.unique(ex.index.normalize()))
    rows, keep = [], {}
    for dows, k, start, hold in itertools.product([(4,), (0, 1, 2, 3)], [2.0, 4.0, 8.0], [1385], [50, 40]):
        dd = days[days.dayofweek.isin(dows)]
        t = dd + pd.Timedelta(minutes=start)
        s = pd.DataFrame({"dir": 1, "sl": k * a15.asof(t - pd.Timedelta(minutes=15)).to_numpy(), "tp": 1e6,
                          "hold_min": hold if start + hold <= 1437 else 1437 - start, "be": 0.0, "trail": 0.0,
                          "leg": "fri_close"}, index=t).dropna()
        res = engine.run(x, s, g)
        tr = res.trades
        m = metrics.summarize(res, "")
        yr = tr.groupby(tr.entry_time.dt.year).R.sum()
        name = f"d{''.join(map(str, dows))}_k{k}_s{start}_h{hold}"
        rows.append({"cfg": name, "n": len(tr), "wr": m["win_rate"], "avgR": m["avg_R"],
                     "t": round(tr.R.mean() / tr.R.std() * np.sqrt(len(tr)), 2), "R_yr": round(tr.R.sum() / 7.7, 1),
                     "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "yrs_pos": int((yr > 0).sum()),
                     **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
    T = pd.DataFrame(rows).sort_values("t", ascending=False)
    (lab.REPORTS / "EXP-036").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-036" / "friday_close.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    top = list(T.cfg.head(2))
    lab.save_experiment("EXP-036", {"strategy": "Friday close drift"}, {c: keep[c][0] for c in top},
                        {c: keep[c][1] for c in top})

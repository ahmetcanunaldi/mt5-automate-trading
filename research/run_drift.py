"""EXP-029: Asia-open drift (and Friday-close drift) as executable intraday strategies, M1 2019-2026, $100k.
Long at ENTRY (server) on selected weekdays, exit after HOLD minutes (or SL = k x ATR_H1(14)), costs incl.
real bar spreads (floor 15 pts), 5-pt slippage each side, $7/lot."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def load():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    h1 = m1.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a_h1 = atr(h1, 14)
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    return m1, a_h1, engine.prepare_exec(ex, 1, blk, flt)


def drift_signals(m1, a_h1, entry_min=65, hold=115, days=(1, 2, 3, 4), sl_k=3.0, direction=1):
    days_idx = pd.DatetimeIndex(np.unique(m1.loc[A:B].index.normalize()))
    days_idx = days_idx[days_idx.dayofweek.isin(days)]
    t = days_idx + pd.Timedelta(minutes=entry_min)
    a = a_h1.asof(t - pd.Timedelta(hours=1)).to_numpy()        # last CLOSED H1 bar's ATR
    s = pd.DataFrame({"dir": direction, "sl": sl_k * a, "tp": 1e6 * np.ones(len(t)), "hold_min": hold, "be": 0.0,
                      "trail": 0.0, "leg": "drift"}, index=t)
    return s[np.isfinite(s.sl) & (s.sl > 0)]


if __name__ == "__main__":
    m1, a_h1, x = load()
    g = engine.Guards(initial_balance=100_000, max_trades_day=3, total_stop_pct=100.0, total_derisk_pct=100.0)
    rows, keep = [], {}
    grid = itertools.product([65, 75], [55, 115, 235, 475], [(0, 1, 2, 3, 4), (1, 2, 3, 4), (3,)], [2.0, 4.0])
    for em, hold, days, k in grid:
        s = drift_signals(m1, a_h1, em, hold, days, k)
        res = engine.run(x, s, g)
        t = res.trades
        m = metrics.summarize(res, "")
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"e{em}_h{hold}_d{''.join(map(str, days))}_k{k}"
        rows.append({"cfg": name, "n": len(t), "wr": m["win_rate"], "avgR": m["avg_R"],
                     "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "SR": m["sharpe"], "net%": m["net_pct"],
                     "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"], "yrs_pos": int((yr > 0).sum()),
                     **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("SR", ascending=False)
    (lab.REPORTS / "EXP-029").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-029" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    top = list(T.cfg.head(3))
    lab.save_experiment("EXP-029", {"strategy": "Asia-open drift long"}, {c: keep[c][0] for c in top},
                        {c: keep[c][1] for c in top})

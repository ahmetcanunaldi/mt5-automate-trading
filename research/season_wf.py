"""EXP-036: walk-forward seasonal schedule. Select weekday x hour x hold cells on 2019-2022 (|t| > 2.5,
yrs_same >= 0.75) and trade them (direction = sign of the in-sample mean) on 2023-2026 with the engine
(M1 exec, real costs, $100k, SL 3 ATR_H1). Entry at HH:05 to skip the first minutes of each hour."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402


def cell_stats(m1, a, b):
    x = m1.loc[a:b]
    h1 = x["close"].resample("1h").last().dropna()
    op = x["close"].resample("1h").first()  # first M1 CLOSE of the hour ~ HH:00-HH:01 -> use HH:05 below
    c5 = x["close"][x.index.minute == 5]; c5.index = c5.index.floor("1h")
    rows = []
    for hold in (1, 2, 4):
        end = h1.shift(-(hold - 1))
        r = np.log(end / c5.reindex(h1.index)) * 1e4
        same = h1.index.normalize() == pd.Series(h1.index, index=h1.index).shift(-(hold - 1)).dt.normalize()
        f = pd.DataFrame({"r": r, "ok": same.to_numpy()}, index=h1.index)
        f = f[f.ok & f.r.notna()]
        for (dw, hr), g in f.groupby([f.index.dayofweek, f.index.hour]):
            if len(g) < 100:
                continue
            m = g.r.mean(); t = m / g.r.std() * np.sqrt(len(g))
            ym = g.groupby(g.index.year).r.mean()
            rows.append({"dow": dw, "hour": hr, "hold": hold, "bps": m, "t": t,
                         "yrs_same": (np.sign(ym) == np.sign(m)).mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    h1b = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    a_h1 = atr(h1b, 14)
    IS = cell_stats(m1, "2019", "2022")
    OS = cell_stats(m1, "2023", "2026")
    sel = IS[(IS.t.abs() > 2.5) & (IS.yrs_same >= 0.75) & (IS.hour >= 1) & (IS.hour <= 22)]
    sel = sel.sort_values("t", key=abs, ascending=False).drop_duplicates(["dow", "hour"])
    chk = sel.merge(OS, on=["dow", "hour", "hold"], suffixes=("_is", "_oos"))
    chk["same_sign"] = np.sign(chk.bps_is) == np.sign(chk.bps_oos)
    pd.set_option("display.width", 200)
    print("selected cells (2019-22) and their 2023-26 behaviour:\n", chk.round(2).to_string(index=False))
    print("sign kept out-of-sample:", chk.same_sign.mean().round(2), " mean OOS t (signed):",
          (np.sign(chk.bps_is) * chk.t_oos).mean().round(2))
    # trade the schedule in 2023-2026
    ex = m1.loc["2023":"2026-09-26"]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    days = pd.DatetimeIndex(np.unique(ex.index.normalize()))
    parts = []
    for r in sel.itertuples():
        dd = days[days.dayofweek == r.dow]
        t = dd + pd.Timedelta(hours=int(r.hour), minutes=5)
        parts.append(pd.DataFrame({"dir": int(np.sign(r.bps)), "sl": 3.0 * a_h1.asof(t - pd.Timedelta(hours=1)).to_numpy(),
                                   "tp": 1e6, "hold_min": 60 * int(r.hold) - 5, "be": 0.0, "trail": 0.0,
                                   "leg": f"{r.dow}-{r.hour}-{r.hold}"}, index=t))
    s = pd.concat(parts).sort_index().dropna()
    g = engine.Guards(initial_balance=100_000, max_trades_day=10, max_positions=2, max_open_risk_pct=1.0,
                      total_stop_pct=100.0, total_derisk_pct=100.0)
    res = engine.run(x, s, g)
    m = metrics.summarize(res, "season_schedule_OOS_2023-26")
    t = res.trades
    print("\nOOS schedule:", {k: m[k] for k in ("trades", "win_rate", "avg_R", "profit_factor", "sharpe", "net_pct",
                                                "max_total_dd_pct")},
          "by year R:", t.groupby(t.entry_time.dt.year).R.sum().round(1).to_dict())
    lab.save_experiment("EXP-036", {"cells": sel.round(3).to_dict("records")}, {"OOS": m}, {"OOS_2023-26": res})

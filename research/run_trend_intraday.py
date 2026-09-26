"""EXP-034: intraday trades conditioned on the higher-timeframe trend (gold trends on H4/D1, mean-reverts intraday).

Daily trend state (known at the day's open): UP if D1 close > EMA50 and EMA20 > EMA50 (yesterday), DOWN mirror.
All trades are intraday (flat by 23:45 server), M15 decisions, M1 execution, $100k, real guards, 2019-2026.
  dip   : in an UP day, price trades >= k * ATR_D below the day's high (between 03:00 and 20:00) -> long at the
          M15 close, SL = sl * ATR_D below entry, TP = back to the day's high ("high") or rr * risk.
  tday  : at decision minute T, move from the day open >= k * ATR_D in the trend direction and close in the top
          (bottom) 25 % of the day range -> hold to end of day, SL = sl * ATR_D.
  drive : at 03:05, first-2h move (01:05 -> 03:05) in the trend direction >= k * ATR_D -> hold to end of day.
"""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr, ema  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def setup():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"])
    d1 = d1[d1.index.dayofweek < 5]
    st = pd.DataFrame({"atr": atr(d1, 14), "c": d1.close, "e20": ema(d1.close, 20), "e50": ema(d1.close, 50)})
    state = np.where((st.c > st.e50) & (st.e20 > st.e50), 1, np.where((st.c < st.e50) & (st.e20 < st.e50), -1, 0))
    day = pd.DataFrame({"state": pd.Series(state, index=d1.index).shift(1), "atr": st.atr.shift(1)})
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    return m15.loc[A:B], day, engine.prepare_exec(ex, 1, blk, flt)


def scan(m15, day, kind, k, sl, rr=None, T=720, states=(1, -1), tp_mode="rr"):
    rows = []
    for dte, g in m15.groupby(m15.index.normalize()):
        if dte not in day.index or dte.dayofweek > 4:
            continue
        s, a = day.at[dte, "state"], day.at[dte, "atr"]
        if s not in states or not (a > 0):
            continue
        o = g.open.iloc[0]
        hi = g.high.cummax(); lo = g.low.cummin()
        tm = g.index.hour * 60 + g.index.minute
        if kind == "dip":
            cond = ((hi - g.close) >= k * a) if s == 1 else ((g.close - lo) >= k * a)
            cond &= (tm >= 180) & (tm <= 1200)
        elif kind == "tday":
            rng = (hi - lo).replace(0, np.nan)
            pos = (g.close - lo) / rng
            cond = (tm == T) & (((g.close - o) * s) >= k * a) & ((pos >= 0.75) if s == 1 else (pos <= 0.25))
        else:  # drive
            cond = (tm == 180) & (((g.close - o) * s) >= k * a)
        idx = np.flatnonzero(cond.to_numpy())
        if len(idx) == 0:
            continue
        i = idx[0]
        entry = g.close.iloc[i]
        risk = sl * a
        if kind == "dip" and tp_mode == "high":
            tp = (hi.iloc[i] - entry) if s == 1 else (entry - lo.iloc[i])
        elif rr is not None:
            tp = rr * risk
        else:
            tp = 50 * a
        if tp <= 0.2 * risk:
            continue
        rows.append({"t": g.index[i] + pd.Timedelta(minutes=15), "dir": int(s), "sl": risk, "tp": tp})
    if not rows:
        return pd.DataFrame()
    x = pd.DataFrame(rows).set_index("t")
    x["hold_min"] = 1440; x["be"] = 0.0; x["trail"] = 0.0; x["leg"] = kind
    return x


if __name__ == "__main__":
    m15, day, x = setup()
    g = engine.Guards(initial_balance=100_000, max_trades_day=3, total_stop_pct=100.0, total_derisk_pct=100.0)
    rows, keep = [], {}
    cfgs = []
    cfgs += [("dip", k, sl, rr, None, tp) for k, sl, rr, tp in itertools.product([0.3, 0.5, 0.8], [0.5, 1.0], [1.0, 2.0], ["rr", "high"])]
    cfgs += [("tday", k, sl, None, T, "rr") for k, sl, T in itertools.product([0.3, 0.5, 0.8], [0.5, 1.0], [600, 900, 1080])]
    cfgs += [("drive", k, sl, None, None, "rr") for k, sl in itertools.product([0.1, 0.2, 0.4], [0.5, 1.0])]
    for kind, k, sl, rr, T, tpm in cfgs:
        s = scan(m15, day, kind, k, sl, rr, T or 720, tp_mode=tpm)
        if len(s) < 30:
            continue
        res = engine.run(x, s, g)
        t = res.trades
        m = metrics.summarize(res, "")
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"{kind}_k{k}_sl{sl}_rr{rr}_T{T}_{tpm}"
        rows.append({"cfg": name, "n": len(t), "wr": m["win_rate"], "avgR": m["avg_R"],
                     "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "SR": m["sharpe"], "net%": m["net_pct"],
                     "DD%": m["max_total_dd_pct"], "yrs_pos": int((yr > 0).sum()), **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("t", ascending=False)
    (lab.REPORTS / "EXP-034").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-034" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    top = list(T.cfg.head(2))
    lab.save_experiment("EXP-034", {"strategy": "trend-conditioned intraday"}, {c: keep[c][0] for c in top},
                        {c: keep[c][1] for c in top})

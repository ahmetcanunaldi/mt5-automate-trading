"""EXP-067: long breakouts of the previous day's high during Asian hours (01:05-10:00), optionally only after a
strong close; exit at 10:00 or end of day; SL k * ATR_D. Single leg on M1 2019-2026 and added to best12+season."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, metrics  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.combo8 import BEST12, build  # noqa: E402
from research.features import atr  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

if __name__ == "__main__":
    m1, legs = build()
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"]).loc[A:B]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    ad = atr(d1, 14).shift(1)
    clv_prev = (((d1.close - d1.low) - (d1.high - d1.close)) / (d1.high - d1.low)).shift(1)
    pdh = d1.high.shift(1)
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    tm = m15.index.hour * 60 + m15.index.minute
    rows, best = [], None
    for need_sc, exit_at, sl in itertools.product([False, True], ["10:00", "eod"], [0.5, 1.0]):
        sig = []
        for day, gdf in m15[(tm >= 65) & (tm < 600)].groupby(m15[(tm >= 65) & (tm < 600)].index.normalize()):
            if day not in pdh.index or not np.isfinite(pdh[day]) or (need_sc and not clv_prev[day] > 0.3):
                continue
            hit = np.flatnonzero((gdf.close > pdh[day]).to_numpy())
            if len(hit) == 0:
                continue
            i = hit[0]
            t = gdf.index[i] + pd.Timedelta(minutes=15)
            hold = (600 - (t.hour * 60 + t.minute)) if exit_at == "10:00" else (1410 - (t.hour * 60 + t.minute))
            if hold < 15:
                continue
            sig.append({"t": t, "dir": 1, "sl": sl * ad[day], "tp": 1e6, "hold_min": hold, "be": 0.0, "trail": 0.0,
                        "leg": "asia_pdh"})
        s = pd.DataFrame(sig).set_index("t")
        gg = engine.Guards(initial_balance=100_000, max_trades_day=3, total_stop_pct=100.0, total_derisk_pct=100.0)
        tr = engine.run(x, s, gg).trades
        yr = tr.groupby(tr.entry_time.dt.year).R.sum()
        tt = tr.R.mean() / tr.R.std() * np.sqrt(len(tr))
        rows.append({"after_strong_close": need_sc, "exit": exit_at, "sl": sl, "n": len(tr), "avgR": round(tr.R.mean(), 3),
                     "t": round(tt, 2), "R_yr": round(tr.R.sum() / 7.7, 1), "yrs_pos": int((yr > 0).sum())})
        if best is None or tt > best[0]:
            best = (tt, s)
    print(pd.DataFrame(rows).to_string(index=False))
    legs["asia_pdh"] = best[1]
    out = []
    for name, extra in (("best12+season0.5", []), ("+asia_pdh", ["asia_pdh"])):
        parts = [legs[n] for n in BEST12 + extra]
        sz = legs["season"].copy(); sz["risk_mult"] = 0.5; parts.append(sz)
        res = engine.run(x, pd.concat(parts).sort_index(kind="stable"), g(True, 6))
        m = metrics.summarize(res, name); ch = metrics.challenge_sim(res, max_days=250)
        out.append({"book": name, "R_yr": round(res.trades.R.sum() / 7.7, 1), "SR": m["sharpe"], "DD%": m["max_total_dd_pct"],
                    "dDD%": m["max_daily_dd_pct"], "week_R": m["week_R_mean"], "pass250": ch["pass_rate"], "fail": ch["fail_rate"]})
    print(pd.DataFrame(out).to_string(index=False))

"""EXP-057: strong-close continuation leg. If yesterday closed in the top q of its range (and optionally in the
20-day high zone / in January), go long at today's 01:05 open, exit at 23:30 (intraday) or hold h days (swing),
SL = s * ATR_D. D1 2008-2026 (engine on daily bars) and M1 2019-2026 (engine on M1). Then portfolio impact."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.features import atr  # noqa: E402
from research.final_candidate import LEGS9, book  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_swing import COSTS, exec_d1, guards  # noqa: E402
from research.run_trend_intraday import scan, setup as td_setup  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402


def signals(d, q=0.8, sl=1.0, zone=None, hold_days=1, t_min=65):
    a = atr(d, 14)
    clv = ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low).replace(0, np.nan)
    cond = clv > q
    if zone:
        hi20, lo20 = d.high.rolling(20).max(), d.low.rolling(20).min()
        cond &= ((d.close - lo20) / (hi20 - lo20)) > zone
    idx = d.index[cond.fillna(False).to_numpy()]
    nxt = d.index.searchsorted(idx, side="right")
    ok = nxt < len(d.index)
    t = d.index[nxt[ok]] + pd.Timedelta(minutes=t_min)
    return pd.DataFrame({"dir": 1, "sl": sl * a.loc[idx[ok]].to_numpy(), "tp": 1e6,
                         "hold_min": 1440 * hold_days - (t_min + 15) if hold_days == 1 else 1440 * hold_days,
                         "be": 0.0, "trail": 0.0, "leg": "strong_close"}, index=t).dropna()


if __name__ == "__main__":
    # --- D1 2008-2026 ---
    d_all = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
    _, xd = exec_d1()
    print("=== D1 2008-2026 (entry next open, exit that close) ===")
    for q, zone, sl in itertools.product([0.6, 0.8], [None, 0.8], [1.0, 1.5]):
        s = signals(d_all, q, sl, zone, t_min=0).loc["2008-02":]
        s["hold_min"] = 1440
        res = engine.run(xd, s, guards(True), COSTS)
        t = res.trades
        er = {f"{a_}-{str(b_)[2:]}": round(t[(t.entry_time.dt.year >= a_) & (t.entry_time.dt.year <= b_)].R.sum() / (b_ - a_ + 1), 1)
              for a_, b_ in ((2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026))}
        print(f"q{q} zone{zone} sl{sl}: n {len(t)} avgR {t.R.mean():+.3f} t {t.R.mean() / t.R.std() * np.sqrt(len(t)):+.2f} "
              f"R/yr by era {er}")
    # --- M1 2019-2026 + portfolio ---
    m1, legs = all_legs()
    m15_td, day, _ = td_setup()
    s = scan(m15_td, day, "tday", 0.3, 1.0, None, 900); s["hold_min"] = 510; s["leg"] = "tday900"
    legs["tday900"] = s.loc[A:B]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    print("\n=== M1 2019-2026 single leg ===")
    best = None
    for q, zone, sl in itertools.product([0.6, 0.8], [None, 0.8], [1.0, 1.5]):
        s = signals(d1, q, sl, zone).loc[A:B]
        g = engine.Guards(initial_balance=100_000, max_trades_day=3, total_stop_pct=100.0, total_derisk_pct=100.0)
        t = engine.run(x, s, g).trades
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        tt = t.R.mean() / t.R.std() * np.sqrt(len(t))
        print(f"q{q} zone{zone} sl{sl}: n {len(t)} avgR {t.R.mean():+.3f} t {tt:+.2f} R/yr {t.R.sum() / 7.7:+.1f} yrs+ {(yr > 0).sum()}/8")
        if best is None or tt > best[0]:
            best = (tt, s)
    legs["strong_close"] = best[1]
    base = LEGS9 + ["tday900"]
    rows = []
    for name, names in (("best10", base), ("best10+strong_close", base + ["strong_close"])):
        r, res, m = book(x, legs, names, True, 6)
        rows.append({"book": name, **r, "week_R": m["week_R_mean"], "wk>=2R": m["weeks_ge_2R"]})
    print(pd.DataFrame(rows).to_string(index=False))

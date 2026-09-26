"""EXP-035: portfolio of the thin-but-consistent edges found so far (2019-2026, M1 execution, $100k, real guards).

Legs (all long-biased except the trend-day leg in down-trends; the engine forbids opposite positions):
  trendH4  : H4 Donchian-180 long, SL 2 ATR_H4, trail 6 ATR_H4, max hold 240 H4 bars      (EXP-033, swing)
  tday     : 18:00 server trend-day continuation k0.3, SL 1 ATR_D, hold to 23:30          (EXP-034, intraday)
  drift    : Asia-open drift long Tue-Fri 01:15 -> +475 min, SL 4 ATR_H1                  (EXP-029, intraday)
  friday   : Friday long 01:05 -> 23:00, SL 1.5 ATR_D                                      (EXP-027, intraday)
  tom      : turn-of-month long, first 3 trading days of the month, SL 2 ATR_D           (EXP-027, swing)
"""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_drift import drift_signals  # noqa: E402
from research.run_trend_intraday import scan  # noqa: E402
from research.strategies import swing  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def build_legs():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    h4 = m1.resample("4h").agg(AGG).dropna(subset=["open"])
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    atr_d = atr(d1, 14).shift(1)
    legs = {}
    df4 = swing.prep(h4, bar=pd.Timedelta(hours=4))
    legs["trendH4"] = swing.donchian(df4, n=180, sl=2.0, trail=6.0, tp=50.0, hold=240, both=False)
    # trend-day continuation (needs the daily state frame used in EXP-034)
    from research.run_trend_intraday import setup as td_setup
    m15_td, day, _ = td_setup()
    s = scan(m15_td, day, "tday", 0.3, 1.0, None, 1080)
    s["hold_min"] = 330
    legs["tday"] = s
    legs["drift"] = drift_signals(m1, atr(h1, 14), entry_min=75, hold=475, days=(1, 2, 3, 4), sl_k=4.0)
    days = pd.DatetimeIndex(np.unique(m1.loc[A:B].index.normalize()))
    fri = days[days.dayofweek == 4]
    legs["friday"] = pd.DataFrame({"dir": 1, "sl": 1.5 * atr_d.reindex(fri).to_numpy(), "tp": 1e6, "hold_min": 1315,
                                   "be": 0.0, "trail": 0.0}, index=fri + pd.Timedelta(minutes=65)).dropna()
    first3 = days[pd.Series(days, index=days).groupby(days.to_period("M")).cumcount().to_numpy() == 0]
    legs["tom"] = pd.DataFrame({"dir": 1, "sl": 2.0 * atr_d.reindex(first3).to_numpy(), "tp": 1e6,
                                "hold_min": 3 * 1440 - 60, "be": 0.0, "trail": 0.0},
                               index=first3 + pd.Timedelta(minutes=70)).dropna()
    for k, v in legs.items():
        v["leg"] = k
        legs[k] = v.loc[A:B]
    return m1, legs


def run_combo(x, legs, names, K=4, wf=False, stop=8.0):
    s = pd.concat([legs[n] for n in names]).sort_index(kind="stable")
    g = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=wf, first_entry_min=65,
                      last_entry_min=1380, max_trades_day=6, max_positions=K, max_open_risk_pct=0.5 * K,
                      total_stop_pct=stop, total_derisk_pct=6.5 if stop < 50 else stop)
    return engine.run(x, s, g)


if __name__ == "__main__":
    m1, legs = build_legs()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows, keep = [], {}
    combos = [["trendH4"], ["tday"], ["drift"], ["friday"], ["tom"],
              ["trendH4", "tday"], ["trendH4", "tday", "drift"], ["trendH4", "tday", "friday", "tom"],
              ["trendH4", "tday", "drift", "friday", "tom"], ["tday", "drift", "friday"]]
    for names, K, wf in itertools.product(combos, [2, 4], [False, True]):
        if len(names) == 1 and K == 4:
            continue
        res = run_combo(x, legs, names, K, wf)
        t = res.trades
        m = metrics.summarize(res, "+".join(names))
        m.update(metrics.monte_carlo_dd(res))
        ch = metrics.challenge_sim(res, max_days=250)
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"{'+'.join(names)}|K{K}|wf{int(wf)}"
        rows.append({"cfg": name, "n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / 7.7, 1), "SR": m["sharpe"],
                     "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
                     "mc95": m["mc_dd95_pct"], "pass250": ch["pass_rate"], "fail": ch["fail_rate"],
                     "med_days": ch["median_days_to_pass"], "yrs_pos": int((yr > 0).sum())})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("SR", ascending=False)
    (lab.REPORTS / "EXP-035").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-035" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    top = list(T.cfg.head(3))
    lab.save_experiment("EXP-035", {"legs": list(legs)}, {c: keep[c][0] for c in top},
                        {c.replace("|", "_").replace("+", "-"): keep[c][1] for c in top})

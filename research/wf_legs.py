"""EXP-040: walk-forward LEG SELECTION to remove the in-sample selection bias of EXP-038.

Pool = every strategy family built so far (winners AND losers). Each leg is run alone (2019-2026, M1 exec,
$100k, K1, no total stop) to obtain its yearly R. At the start of year Y (2021..2026) a leg is selected if its
trailing 2-year R sum > 0 AND trailing t-stat > 1. The out-of-sample book of year Y = sum of the selected legs'
year-Y R (approximation: ignores portfolio-level caps). Compared with "all legs" and with the EXP-038 hand pick.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, features, lab  # noqa: E402
from research.combo import A, B, build_legs  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.oos import CANDS  # noqa: E402,F401
from research.portfolio import LEGS, leg_signals  # noqa: E402
from research.run_volbreak import daily_ctx, signals as vb_signals  # noqa: E402
from research.strategies import swing  # noqa: E402
from research.strategies.amd import amd_signals  # noqa: E402

if __name__ == "__main__":
    m1, legs = build_legs()
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    h4 = m1.resample("4h").agg(AGG).dropna(subset=["open"])
    ctx = daily_ctx(m1)
    legs["lw"] = vb_signals(m15.loc[A:B], ctx, "lw", k=0.4, sl=1.0, start_min=600)
    legs["inside"] = vb_signals(m15.loc[A:B], ctx, "nr", pattern="inside", trend=True)
    legs["nr7"] = vb_signals(m15.loc[A:B], ctx, "nr", pattern="nr7", trend=True)
    a15 = atr(m15, 14)
    days = pd.DatetimeIndex(np.unique(m1.loc[A:B].index.normalize()))
    fri = days[days.dayofweek == 4] + pd.Timedelta(minutes=1385)
    legs["fri_close"] = pd.DataFrame({"dir": 1, "sl": 4.0 * a15.asof(fri - pd.Timedelta(minutes=15)).to_numpy(),
                                      "tp": 1e6, "hold_min": 50, "be": 0.0, "trail": 0.0}, index=fri).dropna()
    # losers / neutral families (so the selector has something to reject)
    df4 = swing.prep(h4, bar=pd.Timedelta(hours=4))
    legs["h4_rsi2"] = swing.rsi2(df4, lo=10, hi=90, sl=2.0, tp=1.0, hold=6)
    legs["h4_bbmr"] = swing.bollinger_mr(df4, sl=2.0, tp=1.0, hold=6)
    legs["amd_lny"] = amd_signals(m15, "london_ny", sweep_atr=0.5, confirm_bars=2, tp_mode="range")
    legs["amd_asia"] = amd_signals(m15, "asia_london", sweep_atr=0.25, confirm_bars=4, tp_mode="range")
    dfm = features.base_frame(m15, h1)
    for n in ("ny_orb", "donchian_us", "pdb_long"):
        legs[f"old_{n}"] = leg_signals(dfm, [n])
    for k in list(legs):
        legs[k] = legs[k].loc[A:B].copy(); legs[k]["leg"] = k
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    swing_legs = {"trendH4", "tom", "h4_rsi2", "h4_bbmr"}
    yearly = {}
    for k, s in legs.items():
        g = engine.Guards(initial_balance=100_000, intraday=k not in swing_legs, weekend_flat=False,
                          last_entry_min=1390, flatten_min=1437 if k == "fri_close" else 1425,
                          fri_flatten_min=1500 if k == "fri_close" else 1350, max_trades_day=5,
                          total_stop_pct=100.0, total_derisk_pct=100.0)
        t = engine.run(x, s, g).trades
        yearly[k] = t.groupby(t.entry_time.dt.year).R.agg(["sum", "count", "std"])
        print(f"{k:<14} n {len(t):>5}  R/yr {t.R.sum() / 7.7:+6.1f}  by year {t.groupby(t.entry_time.dt.year).R.sum().round(1).to_dict()}", flush=True)
    Y = pd.DataFrame({k: v["sum"] for k, v in yearly.items()}).fillna(0.0)
    N = pd.DataFrame({k: v["count"] for k, v in yearly.items()}).fillna(0)
    S = pd.DataFrame({k: v["std"] for k, v in yearly.items()}).fillna(1.0)
    rows = []
    hand = ["trendH4", "tday", "drift", "friday", "tom", "inside", "nr7"]
    for y in range(2021, 2027):
        tr = Y.loc[y - 2:y - 1]
        tstat = tr.sum() / np.sqrt((S.loc[y - 2:y - 1] ** 2 * N.loc[y - 2:y - 1]).sum().replace(0, np.nan))
        sel = [k for k in Y.columns if tr[k].sum() > 0 and tstat[k] > 1]
        rows.append({"year": y, "n_sel": len(sel), "WF_R": round(Y.loc[y, sel].sum(), 1),
                     "ALL_R": round(Y.loc[y].sum(), 1), "HAND_R": round(Y.loc[y, hand].sum(), 1),
                     "selected": ",".join(sel)})
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print(T.to_string(index=False))
    print("total 2021-26  WF:", T.WF_R.sum().round(1), " ALL:", T.ALL_R.sum().round(1), " HAND:", T.HAND_R.sum().round(1))
    (lab.REPORTS / "EXP-040").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-040" / "wf_selection.csv", index=False)
    Y.to_csv(lab.REPORTS / "EXP-040" / "leg_yearly_R.csv")

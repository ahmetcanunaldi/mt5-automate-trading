"""EXP-084: executable index legs (NAS100 / DJ30 / GER40), M1 2019-2026, standalone, new news rule, index costs.

Priors from the equity literature, confirmed on D1 2013-26 (EXP-083):
  mon        : Monday long 01:05 -> 23:30, SL 1.5 ATR_D                                  (weekend effect reversal)
  dip_clv    : yesterday closed in the bottom of its range (clv < -0.6) -> long 01:05 -> 23:30, SL 1 ATR_D
  dip_low20  : yesterday's close in the lowest 10 % of the 20-day range -> long, same exits (short-term reversal)
  tom        : long on the last trading day of the month 01:05, hold 4 trading days (flat Friday), SL 2 ATR_D
  prefomc    : long 24 h before the FOMC decision, flattened 10 min before it by the news rule, SL 1.5 ATR_D
  hi20       : close in the top 10 % of the 20-day range -> long next day (momentum), SL 1 ATR_D
  up3        : 3+ up days -> long next day, SL 1 ATR_D
plus the XAUUSD book's legs (research/multi_legs.legs_for) and the swing families (research/fx_swing.FAM)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, symbols  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.fx_swing import FAM  # noqa: E402
from research.multi_legs import legs_for  # noqa: E402
from research.multi_scan import tstat  # noqa: E402
from research.strategies import swing  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def _mk(times, sl, hold, leg, d=1):
    s = pd.DataFrame({"dir": d, "sl": sl, "tp": 1e6, "hold_min": hold, "be": 0.0, "trail": 0.0, "leg": leg}, index=times)
    return s[np.isfinite(s.sl) & (s.sl > 0)]


SESSION = {"NAS100": (65, 1410), "DJ30": (65, 1410), "GER40": (185, 1370)}   # entry / exit minute (server)


def next_day_signals(d1, cond, leg, sl_k=1.0, t_min=65, t_exit=1410):
    a = atr(d1, 14)
    idx = d1.index[cond.fillna(False).to_numpy(bool)]
    nxt = d1.index.searchsorted(idx, side="right"); ok = nxt < len(d1)
    return _mk(d1.index[nxt[ok]] + pd.Timedelta(minutes=t_min), sl_k * a.loc[idx[ok]].to_numpy(), t_exit - t_min, leg)


def index_legs(m1, sym="NAS100"):
    t0, t1 = SESSION[sym]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    a_prev = atr(d1, 14).shift(1)
    days = d1.loc[A:B].index
    L = {}
    mon = days[days.dayofweek == 0]
    L["mon"] = _mk(mon + pd.Timedelta(minutes=t0), 1.5 * a_prev.reindex(mon).to_numpy(), t1 - t0, "mon")
    clv = ((d1.close - d1.low) - (d1.high - d1.close)) / (d1.high - d1.low).replace(0, np.nan)
    pos20 = (d1.close - d1.low.rolling(20).min()) / (d1.high.rolling(20).max() - d1.low.rolling(20).min())
    up = (d1.close > d1.close.shift(1)).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    L["dip_clv"] = next_day_signals(d1, clv < -0.6, "dip_clv", t_min=t0, t_exit=t1)
    L["dip_low20"] = next_day_signals(d1, pos20 < 0.1, "dip_low20", t_min=t0, t_exit=t1)
    L["hi20"] = next_day_signals(d1, pos20 > 0.9, "hi20", t_min=t0, t_exit=t1)
    L["up3"] = next_day_signals(d1, (streak >= 3) & (up == 1), "up3", t_min=t0, t_exit=t1)
    per = days.to_period("M")
    last = days[pd.Series(1, index=days).groupby(per).cumcount(ascending=False).to_numpy() == 0]
    L["tom"] = _mk(last + pd.Timedelta(minutes=t0), 2.0 * a_prev.reindex(last).to_numpy(), 4 * 1440, "tom")
    cal = pd.read_csv(calendar_news.CAL_PATH)
    f = cal[cal.event_code == "fed-interest-rate-decision"]
    ft = calendar_news.utc_to_server(pd.DatetimeIndex(pd.to_datetime(f.time_server) - pd.Timedelta(hours=3)))
    ft = ft[(ft >= pd.Timestamp(A)) & (ft <= pd.Timestamp(B))]
    ent = ft - pd.Timedelta(hours=24)
    ent = ent.where(ent.dayofweek < 5, ent - pd.Timedelta(days=2))
    L["prefomc"] = _mk(ent, 1.5 * a_prev.asof(ent).to_numpy(), 1500, "prefomc")
    for k in L:
        L[k] = L[k].loc[A:B].copy(); L[k]["trail"] = 1.5 * L[k].sl if k != "tom" else 0.0
    return L


def gidx():
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                         flatten_min=1425, fri_flatten_min=1350, max_trades_day=8, max_positions=1,
                         max_open_risk_pct=0.5, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0)


if __name__ == "__main__":
    rows = []
    for sym in ("NAS100", "DJ30", "GER40"):
        m1 = symbols.load_m1(sym)
        x = symbols.prepare(sym, m1.loc[A:B])
        L = index_legs(m1, sym)
        L.update({f"xau_{k}": v for k, v in legs_for(m1).items()})
        d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
        df = swing.prep(d1, bar=pd.Timedelta(days=1))
        for fam, fn in FAM.items():
            s = fn(df).loc[A:B]
            wk = s.index.dayofweek >= 5
            s.index = s.index + pd.to_timedelta(np.where(wk, 7 - s.index.dayofweek, 0), unit="D") + pd.Timedelta(minutes=65)
            for dd, tag in ((1, "L"), (-1, "S")):
                sd = s[s.dir == dd].copy(); sd["leg"] = f"sw_{fam}_{tag}"
                if len(sd) >= 20:
                    L[f"sw_{fam}_{tag}"] = sd
        pd.to_pickle(L, lab.DATA / f"legs_{sym}.pkl")
        for name, s in L.items():
            t = engine.run(x, s, gidx(), symbols.COSTS[sym]).trades
            if len(t) < 15:
                continue
            yr = t.groupby(t.entry_time.dt.year).R.sum()
            rows.append({"sym": sym, "leg": name, "n": len(t), "avgR": round(t.R.mean(), 3), "t": tstat(t.R),
                         "R_yr": round(t.R.sum() / 7.7, 1), "t_19_22": tstat(t[t.entry_time.dt.year <= 2022].R),
                         "t_23_26": tstat(t[t.entry_time.dt.year >= 2023].R), "yrs_pos": int((yr > 0).sum())})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    (lab.REPORTS / "EXP-084").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-084" / "index_legs.csv", index=False)
    pd.set_option("display.width", 220)
    for s, g in T.groupby("sym"):
        print(f"\n=== {s}")
        print(g.sort_values("t", ascending=False).head(22).to_string(index=False))

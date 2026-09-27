"""EXP-094: equity-calendar structures with strong literature priors, on NAS100 / DJ30 / GER40 (and XAU for
reference). D1 2013-2026 (next-day open->close, t by eras) and M1 2019-2026 executable legs.
  preholiday : trading day before an NYSE holiday -> long (Ariel 1990, Kim & Park 1994)
  opex_week  : week of the monthly options expiration (3rd Friday) -> long; week after -> short / avoid
  overnight  : long 22:50 -> next day 16:20 server (US close -> before the US open; Cliff/Cooper/Gulen 2008)
  nfp / cpi  : US employment / CPI release day -> long 01:05-23:30 (entries are blocked around the release
               itself by the news rule, so the position is opened at 01:05 and closed 10 min before)"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,  # noqa: E402
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday)

from research import calendar_news, engine, lab, symbols  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.multi_scan import tstat  # noqa: E402


class NYSE(AbstractHolidayCalendar):
    rules = [Holiday("NewYear", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr, USPresidentsDay,
             GoodFriday, USMemorialDay, Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("July4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


HOL = NYSE().holidays("2012-01-01", "2027-12-31")
A, B = "2019-01-01", "2026-09-26"


def day_flags(idx):
    idx = pd.DatetimeIndex(idx)
    nxt_bd = pd.Series(idx, index=idx).shift(-1)
    pre = pd.Series([any((h > t) and (h < n) for h in HOL[(HOL > t) & (HOL <= t + pd.Timedelta(days=5))]) if pd.notna(n) else False
                     for t, n in zip(idx, nxt_bd)], index=idx)
    pre |= pd.Series(idx.isin(HOL - pd.Timedelta(days=1)), index=idx)
    third_fri = {}
    for per in pd.period_range(idx.min(), idx.max(), freq="M"):
        f = pd.Timestamp(per.start_time); f += pd.Timedelta(days=(4 - f.dayofweek) % 7 + 14); third_fri[per] = f
    tf = pd.Series([third_fri[p] for p in idx.to_period("M")], index=idx)
    opex_wk = (idx >= tf - pd.Timedelta(days=4)) & (idx <= tf)
    after_wk = (idx > tf) & (idx <= tf + pd.Timedelta(days=7))
    return pd.DataFrame({"preholiday": pre.to_numpy(), "opex_week": opex_wk, "after_opex": after_wk}, index=idx)


def d1_tests(sym, f):
    d = pd.read_parquet(lab.DATA / f"{f}.parquet")[["open", "high", "low", "close"]]
    d = d[(d.index.dayofweek < 5)].loc["2013":]
    r = np.log(d.close / d.open) * 1e4
    F = day_flags(d.index)
    rows = []
    for k in F.columns:
        v = r[F[k].to_numpy()]
        eras = [v[(v.index.year >= a) & (v.index.year <= b)].mean() for a, b in ((2013, 2016), (2017, 2019), (2020, 2022), (2023, 2026))]
        rows.append({"sym": sym, "test": f"{k} (same-day open->close)", "n": len(v), "bps": round(v.mean(), 1),
                     "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2), "all_days_bps": round(r.mean(), 1),
                     "eras": [round(e, 1) for e in eras]})
    return rows


def legs(sym, m1):
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    a = atr(d1, 14).shift(1)
    days = d1.loc[A:B].index
    F = day_flags(days)
    mk = lambda ds, t0, hold, k, leg: pd.DataFrame({"dir": 1, "sl": k * a.reindex(ds).to_numpy(), "tp": 1e6, "hold_min": hold,  # noqa: E731
                                                   "be": 0.0, "trail": 1.5 * k * a.reindex(ds).to_numpy(), "leg": leg},
                                                  index=ds + pd.Timedelta(minutes=t0)).dropna()
    L = {"preholiday": mk(days[F.preholiday.to_numpy()], 65, 1345, 1.0, "preholiday"),
         "opex_week": mk(days[F.opex_week.to_numpy()], 65, 1345, 1.0, "opex_week"),
         "after_opex_short": mk(days[F.after_opex.to_numpy()], 65, 1345, 1.0, "after_opex_short").assign(dir=-1),
         "overnight": mk(days[days.dayofweek < 4], 1370, 1050, 1.0, "overnight")}
    cal = pd.read_csv(calendar_news.CAL_PATH)
    for code, name in (("nonfarm-payrolls", "nfp_day"), ("consumer-price-index-mm", "cpi_day")):
        t = calendar_news.utc_to_server(pd.DatetimeIndex(pd.to_datetime(cal[cal.event_code == code].time_server) - pd.Timedelta(hours=3)))
        ds = pd.DatetimeIndex(t.normalize().unique()).intersection(days)
        L[name] = mk(ds, 65, 1345, 1.0, name)
    return L


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    rows = []
    for sym, f in (("NAS100", "NAS100_D1_long"), ("DJ30", "DJ30_D1_long"), ("GER40", "GER40_D1_long"), ("XAUUSD", "XAUUSD_D1_2007")):
        rows += d1_tests(sym, f)
    print(pd.DataFrame(rows).to_string(index=False))
    out = []
    for sym in ("NAS100", "DJ30", "GER40", "XAUUSD"):
        m1 = symbols.load_m1(sym)
        x = symbols.prepare(sym, m1.loc[A:B])
        g1 = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                           max_trades_day=8, max_positions=1, max_open_risk_pct=0.5, risk_on_initial=True,
                           total_stop_pct=100.0, total_derisk_pct=100.0)
        L = legs(sym, m1)
        pd.to_pickle(L, lab.DATA / f"legs_cal_{sym}.pkl")
        for name, s in L.items():
            t = engine.run(x, s, g1, symbols.COSTS[sym]).trades
            if len(t) < 15:
                continue
            yr = t.groupby(t.entry_time.dt.year).R.sum()
            out.append({"sym": sym, "leg": name, "n": len(t), "avgR": round(t.R.mean(), 3), "t": tstat(t.R),
                        "R_yr": round(t.R.sum() / 7.7, 1), "t_19_22": tstat(t[t.entry_time.dt.year <= 2022].R),
                        "t_23_26": tstat(t[t.entry_time.dt.year >= 2023].R), "yrs_pos": int((yr > 0).sum())})
    T = pd.DataFrame(out)
    print(T.to_string(index=False))
    (lab.REPORTS / "EXP-094").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-094" / "legs.csv", index=False)

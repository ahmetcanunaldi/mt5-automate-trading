"""EXP-083: structure of NAS100 / DJ30 / GER40 (Vantage CFDs), for combination with XAUUSD.

(a) daily battery (research/daily_multi.battery) on D1 2013-2026
(b) session decomposition on M1 2019-2026 (server time): Asia 01:05-10:00, Europe 10:00-16:30, US cash open->close
    16:30-23:00, post 23:00-23:55 and overnight (23:00 -> next 16:30); bps/day, t, share of years positive
(c) turn of month (last trading day + first 3), US cash-session hours and full day
(d) intraday momentum (Gao et al. 2018): first 30 min of the US cash session (16:30-17:00 server) -> last 30 min
    (22:30-23:00): signed return
(e) pre-FOMC drift: 24 h before the FOMC decision (MT5 calendar fed-interest-rate-decision)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, lab, symbols  # noqa: E402
from research.daily_multi import battery  # noqa: E402


def at(c, times):
    return c.asof(times).to_numpy()


def stat(label, v):
    v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
    yr = v.groupby(v.index.year).mean()
    return {"test": label, "n": len(v), "bps": round(v.mean(), 2), "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2),
            "yrs_pos": f"{int((yr > 0).sum())}/{len(yr)}", "bps_19_22": round(v[v.index.year <= 2022].mean(), 2),
            "bps_23_26": round(v[v.index.year >= 2023].mean(), 2)}


def sessions(sym, m1):
    c = m1.close
    days = pd.DatetimeIndex(np.unique(m1.index.normalize())); days = days[days.dayofweek < 5]
    T = lambda h, m=0: days + pd.Timedelta(hours=h, minutes=m)  # noqa: E731
    r = lambda a, b: pd.Series(np.log(at(c, b) / at(c, a)) * 1e4, index=days)  # noqa: E731
    rows = [stat("Asia 01:05-10:00", r(T(1, 5), T(10))), stat("Europe 10:00-16:30", r(T(10), T(16, 30))),
            stat("US cash 16:30-23:00", r(T(16, 30), T(23))), stat("post 23:00-23:55", r(T(23), T(23, 55))),
            stat("full 01:05-23:55", r(T(1, 5), T(23, 55)))]
    ov = pd.Series(np.log(at(c, T(16, 30)[1:]) / at(c, T(23)[:-1])) * 1e4, index=days[1:])
    rows.append(stat("overnight 23:00->16:30", ov))
    # turn of month
    per = days.to_period("M")
    fpos = pd.Series(1, index=days).groupby(per).cumcount()
    bpos = pd.Series(1, index=days).groupby(per).cumcount(ascending=False)
    full = r(T(1, 5), T(23, 55))
    rows.append(stat("TOM days (-1..+3) full day", full[(bpos == 0).to_numpy() | (fpos <= 2).to_numpy()]))
    rows.append(stat("non-TOM days full day", full[~((bpos == 0).to_numpy() | (fpos <= 2).to_numpy())]))
    for dw in range(5):
        rows.append(stat(f"weekday {dw} full day", full[days.dayofweek == dw]))
    # intraday momentum
    first = r(T(16, 30), T(17)); last = r(T(22, 30), T(23))
    rows.append(stat("IM: sign(first 30m) x last 30m", np.sign(first) * last))
    prev_ov = pd.Series(np.r_[np.nan, ov.to_numpy()], index=days)
    rows.append(stat("IM: sign(overnight+first30) x last 30m", np.sign(first + prev_ov.fillna(0)) * last))
    # pre-FOMC
    cal = pd.read_csv(calendar_news.CAL_PATH)
    fomc = cal[cal.event_code == "fed-interest-rate-decision"]
    ft = calendar_news.utc_to_server(pd.DatetimeIndex(pd.to_datetime(fomc.time_server) - pd.Timedelta(hours=3)))
    ft = ft[(ft > m1.index[0] + pd.Timedelta(days=2)) & (ft < m1.index[-1])]
    pre = pd.Series(np.log(at(c, ft - pd.Timedelta(minutes=15)) / at(c, ft - pd.Timedelta(hours=24))) * 1e4, index=ft)
    rows.append(stat("pre-FOMC 24h -> 15 min before", pre))
    out = pd.DataFrame(rows); out.insert(0, "sym", sym)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 230)
    d = lab.REPORTS / "EXP-083"; d.mkdir(exist_ok=True)
    B = []
    for sym in ("NAS100", "DJ30", "GER40"):
        B += battery(sym)
    Bt = pd.DataFrame(B); Bt.to_csv(d / "daily_battery.csv", index=False)
    for s, g in Bt.groupby("sym"):
        good = g[(g.t > 2.0) & (g.eras_pos >= 3)].sort_values("t", ascending=False)
        print(f"\n=== {s} daily battery (t > 2, >= 3/4 eras): {len(good)} of {len(g)}")
        print(good.head(12).to_string(index=False))
    S = pd.concat([sessions(s, symbols.load_m1(s).loc["2019":]) for s in ("NAS100", "DJ30", "GER40")])
    S.to_csv(d / "sessions.csv", index=False)
    print("\n", S.to_string(index=False))

"""EXP-055: institutional clock events on M1 2019-2026 (server time; local times converted with real DST rules).

  LBMA gold AM fix 10:30 London, PM fix 15:00 London
  COMEX gold floor/open 08:20 New York, COMEX settlement/close 13:30 New York
  Shanghai Gold Exchange: day-session open 09:00 Shanghai, SGE PM benchmark 15:00 Shanghai (UTC+8, no DST)
For each event: returns in windows relative to the event (bps), t-stat, share of years with the same sign,
and the break-even comparison with the ~2 bps round-trip cost at $2k gold (~1 bps at $4k).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, lab  # noqa: E402

EVENTS = {
    "LBMA_AM_fix": ("Europe/London", "10:30"),
    "LBMA_PM_fix": ("Europe/London", "15:00"),
    "COMEX_open": ("America/New_York", "08:20"),
    "COMEX_close": ("America/New_York", "13:30"),
    "SGE_open": ("Asia/Shanghai", "09:00"),
    "SGE_PM": ("Asia/Shanghai", "15:00"),
    "LDN_open": ("Europe/London", "08:00"),
    "NY_equity_open": ("America/New_York", "09:30"),
}
WINDOWS = [(-60, -5), (-30, 0), (0, 15), (0, 60), (15, 120), (60, 240)]


def server_times(days, tz, hhmm):
    loc = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {hhmm}").tz_localize(tz) for d in days])
    utc = loc.tz_convert("UTC").tz_localize(None)
    return calendar_news.utc_to_server(utc)


if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2019":]
    c = m1["close"]
    days = pd.DatetimeIndex(np.unique(m1.index.normalize()))
    days = days[days.dayofweek < 5]
    rows = []
    for ev, (tz, hhmm) in EVENTS.items():
        T = server_times(days, tz, hhmm)
        for a, b in WINDOWS:
            p0 = c.asof(T + pd.Timedelta(minutes=a)).to_numpy(); p1 = c.asof(T + pd.Timedelta(minutes=b)).to_numpy()
            r = pd.Series(np.log(p1 / p0) * 1e4, index=T)
            same_day = (T + pd.Timedelta(minutes=a)).normalize() == (T + pd.Timedelta(minutes=b)).normalize()
            r = r[same_day].replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) < 200:
                continue
            yr = r.groupby(r.index.year).mean()
            rows.append({"event": ev, "window": f"{a:+d}..{b:+d}m", "n": len(r), "bps": round(r.mean(), 2),
                         "t": round(r.mean() / r.std() * np.sqrt(len(r)), 2),
                         "yrs_same": round((np.sign(yr) == np.sign(r.mean())).mean(), 2),
                         "bps_2023+": round(r[r.index.year >= 2023].mean(), 2)})
    T = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
    pd.set_option("display.width", 200)
    print(T.head(30).to_string(index=False))
    print(f"\ntests {len(T)}; |t| > 3: {(T.t.abs() > 3).sum()}")
    (lab.REPORTS / "EXP-055").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-055" / "clock_events.csv", index=False)

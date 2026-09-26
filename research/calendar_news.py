"""USD high-impact news calendar -> server-time blackout masks.

data/calendar_usd_high.csv is exported from the MT5 economic calendar (tools/export_calendar.py).
Calendar timestamps are a FIXED UTC+3 (verified: NFP shows 16:30 in winter, 15:30 in summer),
while bar timestamps are trade-server time = UTC+2 (winter) / UTC+3 (US DST, NY-close convention).
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAL_PATH = ROOT / "data" / "calendar_usd_high.csv"


def us_dst(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """True when US daylight saving time is active (2nd Sun Mar 07:00 UTC .. 1st Sun Nov 06:00 UTC)."""
    years = ts_utc.year
    out = np.zeros(len(ts_utc), dtype=bool)
    for y in np.unique(years):
        mar = pd.Timestamp(year=y, month=3, day=1)
        start = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7) + pd.Timedelta(hours=7)
        nov = pd.Timestamp(year=y, month=11, day=1)
        end = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7) + pd.Timedelta(hours=6)
        m = years == y
        out[m] = (ts_utc[m] >= start) & (ts_utc[m] < end)
    return out


def utc_to_server(ts_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    off = np.where(us_dst(ts_utc), 3, 2)
    return ts_utc + pd.to_timedelta(off, unit="h")


def load_news_server_times(path=CAL_PATH) -> pd.DatetimeIndex:
    cal = pd.read_csv(path)
    t_utc = pd.DatetimeIndex(pd.to_datetime(cal["time_server"]) - pd.Timedelta(hours=3))
    return pd.DatetimeIndex(np.unique(utc_to_server(t_utc)))


def blackout_masks(bar_times: pd.DatetimeIndex, bar_minutes: int, news: pd.DatetimeIndex,
                   before_min=30, after_min=30, flatten_before_min=10):
    """Per-bar masks (entries happen at bar open t0, forced exits at bar close t1).
    block_entry[i]: some news N satisfies  -after_min < t0 - N < before_min  (i.e. N in (t0-after, t0+before)).
    flatten[i]:     next news N >= t0 satisfies N - t1 < flatten_before_min + bar_minutes, so closing at t1
                    is the last bar close that still leaves >= flatten_before_min before the release.
    """
    t0 = bar_times.values.astype("datetime64[ns]")
    t1 = t0 + np.timedelta64(bar_minutes, "m")
    n = np.sort(news.values.astype("datetime64[ns]"))
    far_future, far_past = np.datetime64("2100-01-01"), np.datetime64("1900-01-01")
    i_next = np.searchsorted(n, t0, side="left")
    nxt = np.where(i_next < len(n), n[np.minimum(i_next, len(n) - 1)], far_future)
    i_prev = i_next - 1
    prv = np.where(i_prev >= 0, n[np.maximum(i_prev, 0)], far_past)
    m = lambda x: np.timedelta64(x, "m")  # noqa: E731
    block_entry = ((nxt - t0) < m(before_min)) | ((t0 - prv) < m(after_min))
    flatten = (nxt - t1) < m(flatten_before_min + bar_minutes)
    return block_entry, flatten

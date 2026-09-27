"""EXP-061: where does gold's return come from? Session returns (bps) by year, M1 2019-2026 (tradable windows:
first 5 minutes after the daily open skipped). Then the executable version of the strongest session(s)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2019":]
c = m1["close"]
days = pd.DatetimeIndex(np.unique(m1.index.normalize())); days = days[days.dayofweek < 5]
W = {"asia 01:05-10:00": (65, 600), "london 10:00-15:30": (600, 930), "us data 15:30-16:30": (930, 990),
     "ny am 16:30-20:00": (990, 1200), "ny pm 20:00-23:30": (1200, 1410), "early asia 01:05-04:00": (65, 240),
     "late asia 04:00-10:00": (240, 600), "ny late 22:00-23:30": (1320, 1410)}
for name, (a, b) in W.items():
    p0 = c.asof(days + pd.Timedelta(minutes=a)).to_numpy(); p1 = c.asof(days + pd.Timedelta(minutes=b)).to_numpy()
    r = pd.Series(np.log(p1 / p0) * 1e4, index=days).dropna()
    yr = r.groupby(r.index.year).mean().round(1)
    print(f"{name:<24} {r.mean():+6.2f} bps t {r.mean() / r.std() * np.sqrt(len(r)):+.2f}  by year {yr.to_dict()}")

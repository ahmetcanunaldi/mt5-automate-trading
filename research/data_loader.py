"""Pull XAUUSD history from the MT5 terminal into data/*.parquet.

Constraints discovered (see docs/knowhow.md):
- The terminal's "Max bars in chart" = 100k caps copy_rates_* per timeframe:
  H1 from 2007, M15 from 2022-07, M5 from 2025-04, M1 only ~3 months.
- copy_rates_range/from are unreliable outside that window -> use copy_rates_from_pos.
- Ticks are NOT capped: copy_ticks_from works from 2024-12 onwards; we aggregate them to M1.
- Bar times are trade-server time (UTC+2 winter / UTC+3 summer, NY-close convention).
"""
import argparse
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # python311._pth disables cwd on sys.path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TERMINAL = r"D:\copy-trade-web-application\mt5_slave_2\terminal64.exe"
SYMBOL = "XAUUSD"
TF = {"H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15, "M5": mt5.TIMEFRAME_M5, "M1": mt5.TIMEFRAME_M1}


def connect():
    if not mt5.initialize(path=TERMINAL):
        raise RuntimeError(mt5.last_error())


def pull_bars(tf_name):
    rates = None
    for n in (99999, 90000, 60000, 30000):
        rates = mt5.copy_rates_from_pos(SYMBOL, TF[tf_name], 0, n)
        if rates is not None:
            break
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # server time, naive
    df = df.drop(columns=["real_volume"]).set_index("time")
    df.to_parquet(DATA / f"{SYMBOL}_{tf_name}.parquet")
    print(tf_name, len(df), df.index[0], "->", df.index[-1])
    return df


def _ticks_to_m1(t):
    df = pd.DataFrame(t)
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["time"] = pd.to_datetime(df["time_msc"], unit="ms")
    df["spr"] = (df["ask"] - df["bid"]) * 100  # points
    g = df.set_index("time").resample("1min")
    out = pd.DataFrame({
        "open": g["bid"].first(), "high": g["bid"].max(), "low": g["bid"].min(), "close": g["bid"].last(),
        "tick_volume": g["bid"].count(), "spread": g["spr"].mean(), "spread_max": g["spr"].max(),
    }).dropna(subset=["open"])
    return out


def pull_ticks_as_m1(start=dt.datetime(2024, 12, 1), chunk=3_000_000):
    """Walk forward through tick history with copy_ticks_from and store M1 bars built from bid ticks."""
    out_path = DATA / f"{SYMBOL}_M1_from_ticks.parquet"
    parts = []
    cursor = start.replace(tzinfo=dt.timezone.utc)
    last_msc = 0
    while True:
        t = mt5.copy_ticks_from(SYMBOL, cursor, chunk, mt5.COPY_TICKS_ALL)
        if t is None or len(t) == 0:
            break
        t = t[t["time_msc"] > last_msc]
        if len(t) == 0:
            break
        last_msc = int(t["time_msc"][-1])
        parts.append(_ticks_to_m1(t))
        cursor = dt.datetime.fromtimestamp(last_msc / 1000, tz=dt.timezone.utc)
        print("ticks up to", cursor, "bars so far", sum(len(p) for p in parts), flush=True)
        if len(t) < chunk * 0.5:
            break
    m1 = pd.concat(parts)
    # chunk boundaries can split a minute -> merge duplicates
    m1 = m1.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                  "tick_volume": "sum", "spread": "mean", "spread_max": "max"})
    m1.to_parquet(out_path)
    print("M1 from ticks", len(m1), m1.index[0], "->", m1.index[-1])
    return m1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", nargs="*", default=["H1", "M15", "M5", "M1"])
    ap.add_argument("--ticks", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    connect()
    for tf in a.bars:
        pull_bars(tf)
    if a.ticks:
        pull_ticks_as_m1()
    mt5.shutdown()

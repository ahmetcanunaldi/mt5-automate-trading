"""Pull ticks for several symbols and aggregate to M1 bars with microstructure statistics.

Per minute (bid-based): OHLC, n (tick count), up/down (bid upticks/downticks), path (sum |dbid|),
spread mean/max (points), ask-side counts. Stored as data/<SYM>_M1T.parquet (server time).
"""
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import MetaTrader5 as mt5  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.data_loader import DATA, TERMINAL  # noqa: E402

SYMBOLS = ["XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "USDX.r", "SP500.r", "NAS100.r"]


def agg(t, point):
    df = pd.DataFrame(t)
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["time"] = pd.to_datetime(df["time_msc"], unit="ms")
    db = df["bid"].diff().fillna(0.0)
    df["up"] = (db > 0).astype(np.int32)
    df["dn"] = (db < 0).astype(np.int32)
    df["path"] = db.abs()
    df["spr"] = (df["ask"] - df["bid"]) / point
    g = df.set_index("time").resample("1min")
    out = pd.DataFrame({
        "open": g["bid"].first(), "high": g["bid"].max(), "low": g["bid"].min(), "close": g["bid"].last(),
        "n": g["bid"].count(), "up": g["up"].sum(), "dn": g["dn"].sum(), "path": g["path"].sum(),
        "spread": g["spr"].mean(), "spread_max": g["spr"].max(),
    }).dropna(subset=["open"])
    return out


def pull(sym, start=dt.datetime(2024, 12, 1), chunk=3_000_000):
    mt5.symbol_select(sym, True)
    point = mt5.symbol_info(sym).point
    parts, cursor, last = [], start.replace(tzinfo=dt.timezone.utc), 0
    while True:
        t = mt5.copy_ticks_from(sym, cursor, chunk, mt5.COPY_TICKS_ALL)
        if t is None or len(t) == 0:
            break
        t = t[t["time_msc"] > last]
        if len(t) == 0:
            break
        last = int(t["time_msc"][-1])
        parts.append(agg(t, point))
        cursor = dt.datetime.fromtimestamp(last / 1000, tz=dt.timezone.utc)
        if len(t) < chunk * 0.5:
            break
    m = pd.concat(parts)
    m = m.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last", "n": "sum",
                                "up": "sum", "dn": "sum", "path": "sum", "spread": "mean", "spread_max": "max"})
    name = sym.replace(".r", "")
    m.to_parquet(DATA / f"{name}_M1T.parquet")
    print(sym, len(m), m.index[0], "->", m.index[-1], flush=True)


if __name__ == "__main__":
    mt5.initialize(path=TERMINAL)
    for s in (sys.argv[1:] or SYMBOLS):
        try:
            pull(s)
        except Exception as e:  # keep going for other symbols
            print(s, "FAILED", e, flush=True)
    mt5.shutdown()

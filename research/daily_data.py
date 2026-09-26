"""Daily (server-day) XAUUSD bars 2007-2026: broker 'H1' file before 2018-10 is actually D1; after that D1 is
rebuilt from the tester M1 export. Also H4 bars from M1 (2018-10+)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum", "spread": "mean"}
CUT = pd.Timestamp("2018-10-01")


def d1():
    old = lab.load("H1")
    old = old[old.index < CUT]
    old = old[old.index == old.index.normalize()]            # daily rows only
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    new = m1.loc[CUT:].resample("1D").agg(AGG).dropna(subset=["open"])
    d = pd.concat([old[list(AGG)], new]).sort_index()
    d = d[d.index.dayofweek < 5]
    d["spread"] = d["spread"].where(d["spread"] > 0, 30.0)
    return d


def h4():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    return m1.resample("4h").agg(AGG).dropna(subset=["open"])


if __name__ == "__main__":
    d = d1()
    d.to_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
    print(len(d), d.index[0], d.index[-1], d.groupby(d.index.year).size().to_dict())
    h4().to_parquet(lab.DATA / "XAUUSD_H4_2018.parquet")

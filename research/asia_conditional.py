"""EXP-063: what sharpens the Asia drift? Next Asia return (01:05->04:00 and 01:05->10:00, bps) conditioned on
the previous day (close location, return sign/size, NY-PM return, Asia return), the opening gap, weekday.
M1 2019-2026."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.features import atr  # noqa: E402

m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2018-12":]
c = m1["close"]
d1 = m1.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
d1 = d1[d1.index.dayofweek < 5]
days = d1.index
at = lambda mins: pd.Series(c.asof(days + pd.Timedelta(minutes=mins)).to_numpy(), index=days)  # noqa: E731
p105, p400, p1000, p2000, p2330 = at(65), at(240), at(600), at(1200), at(1410)
F = pd.DataFrame(index=days)
F["asia_early"] = np.log(p400 / p105) * 1e4
F["asia"] = np.log(p1000 / p105) * 1e4
prev = d1.shift(1)
F["prev_clv"] = (((prev.close - prev.low) - (prev.high - prev.close)) / (prev.high - prev.low)).to_numpy()
F["prev_ret"] = (np.log(d1.close / d1.close.shift(1)) * 1e4).shift(1)
F["prev_nypm"] = (np.log(p2330 / p2000) * 1e4).shift(1)
F["prev_asia"] = F["asia"].shift(1)
F["gap"] = np.log(p105 / d1.close.shift(1)) * 1e4
F["dow"] = days.dayofweek
F["atr_bps"] = (atr(d1, 14) / d1.close * 1e4).shift(1)
F = F.loc["2019":].dropna()


def show(name, mask):
    for tgt in ("asia_early", "asia"):
        v = F.loc[mask, tgt]
        yr = v.groupby(v.index.year).mean()
        print(f"{name:<36} {tgt:<10} n {len(v):>4} {v.mean():+6.2f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f} yrs+ {(yr > 0).sum()}/{len(yr)}")


show("all", F.index == F.index)
for q in (0.3, 0.6):
    show(f"prev clv > {q}", F.prev_clv > q); show(f"prev clv < -{q}", F.prev_clv < -q)
show("prev day up", F.prev_ret > 0); show("prev day down", F.prev_ret < 0)
show("prev NY-PM up", F.prev_nypm > 0); show("prev NY-PM down", F.prev_nypm < 0)
show("prev Asia up", F.prev_asia > 0); show("prev Asia down", F.prev_asia < 0)
show("gap up (>0)", F.gap > 0); show("gap down (<0)", F.gap < 0)
show("high vol (atr > median)", F.atr_bps > F.atr_bps.median()); show("low vol", F.atr_bps <= F.atr_bps.median())
for dw in range(5):
    show(f"weekday {dw}", F.dow == dw)
show("prev clv > 0.3 & gap >= 0", (F.prev_clv > 0.3) & (F.gap >= 0))

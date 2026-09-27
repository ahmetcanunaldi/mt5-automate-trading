"""EXP-056: daily-bar patterns 2008-2026 (entry at the next open, exit at that day's close -> valid on D1 bars).
Next-day open->close return (bps) conditional on the previous day(s): large range days, big up/down days,
streaks, close location, position vs 20-day high/low, month of year; t-stats and per-era signs."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.features import atr  # noqa: E402

d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
a = atr(d, 20)
nxt = (np.log(d.close / d.open) * 1e4).shift(-1)                 # next day's open->close
ret = np.log(d.close / d.close.shift(1)) * 1e4
rng = (d.high - d.low) / a.shift(1)
clv = ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low).replace(0, np.nan)
up = (ret > 0).astype(int)
streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
streak = streak.where(up == 1, -streak)
hi20 = d.high.rolling(20).max(); lo20 = d.low.rolling(20).min()
pos20 = (d.close - lo20) / (hi20 - lo20)
ERAS = [(2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026)]


def stat(name, cond, sign=1):
    v = (sign * nxt[cond.fillna(False)]).dropna()
    if len(v) < 40:
        return None
    er = {f"{x}-{str(y)[2:]}": round(v[(v.index.year >= x) & (v.index.year <= y)].mean(), 1) for x, y in ERAS}
    return {"condition": name, "n": len(v), "bps": round(v.mean(), 1), "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2),
            "eras_pos": sum(val > 0 for val in er.values()), **er}


rows = [stat("all days (baseline)", pd.Series(True, index=d.index))]
big_move = ret.abs() / a.shift(1) * 1e-4 * d.close.shift(1)
for k in (1.5, 2.0):
    rows.append(stat(f"after range > {k} ATR, follow day direction", rng > k, 1) and
                stat(f"after range > {k} ATR, follow day direction", (rng > k), 1))
    rows.append(stat(f"after up day range > {k} ATR (long)", (rng > k) & (ret > 0)))
    rows.append(stat(f"after down day range > {k} ATR (short)", (rng > k) & (ret < 0), -1))
    rows.append(stat(f"after down day range > {k} ATR (long = fade)", (rng > k) & (ret < 0)))
for s in (3, 4, 5):
    rows.append(stat(f"after {s}+ up days (long)", streak >= s))
    rows.append(stat(f"after {s}+ down days (long = fade)", streak <= -s))
    rows.append(stat(f"after {s}+ down days (short = follow)", streak <= -s, -1))
rows.append(stat("close in top 10 % of range (long)", clv > 0.8))
rows.append(stat("close in bottom 10 % of range (short)", clv < -0.8, -1))
rows.append(stat("close in bottom 10 % of range (long = fade)", clv < -0.8))
rows.append(stat("close at 20-day high zone > 0.9 (long)", pos20 > 0.9))
rows.append(stat("close at 20-day low zone < 0.1 (short)", pos20 < 0.1, -1))
rows.append(stat("close at 20-day low zone < 0.1 (long = fade)", pos20 < 0.1))
for mo in range(1, 13):
    rows.append(stat(f"month {mo:02d} (long)", pd.Series(d.index.month == mo, index=d.index).shift(-1).fillna(False).astype(bool)))
T = pd.DataFrame([r for r in rows if r]).drop_duplicates("condition")
pd.set_option("display.width", 200)
print(T.to_string(index=False))
(lab.REPORTS / "EXP-056").mkdir(exist_ok=True)
T.to_csv(lab.REPORTS / "EXP-056" / "daily_patterns.csv", index=False)

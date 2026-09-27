"""EXP-091: intraday session patterns, M1 2019-2026 (server time; US cash 16:30-23:00, London 10:00).
Indices (NAS100, DJ30, GER40) and XAUUSD. Each test = sign(conditioning move) x later-window return (bps),
i.e. positive = continuation, negative = reversal; plus bucketed versions for gaps.
  on->cash    : overnight (prev 23:00 -> 16:30) vs cash session 16:30->23:00
  eu->cash    : Europe 10:00->16:30 vs cash session
  or30->rest  : first 30 min 16:30->17:00 vs 17:00->23:00
  h1->rest    : first hour vs 17:30->23:00
  day->last   : 01:05->22:00 vs 22:00->23:00
  asia->eu    : 01:05->10:00 vs 10:00->16:30
  gap buckets : overnight move in ATR_D units (< -0.5, -0.5..-0.2, ..., > 0.5) -> first hour 16:30->17:30
                and cash session (fill vs go)
Report: bps, t, years with the same sign, halves agree; |t| > 3 and stable = lead."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab, symbols  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402


def stat(sym, label, v):
    v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 60:
        return None
    yr = v.groupby(v.index.year).mean()
    a, b = v[v.index.year <= 2022].mean(), v[v.index.year >= 2023].mean()
    return {"sym": sym, "test": label, "n": len(v), "bps": round(v.mean(), 2), "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2),
            "yrs_same": round((np.sign(yr) == np.sign(v.mean())).mean(), 2), "h1": round(a, 2), "h2": round(b, 2),
            "halves": bool(np.sign(a) == np.sign(b))}


def run(sym):
    m1 = symbols.load_m1(sym).loc["2019":"2026-09-25"]
    c = m1.close
    days = pd.DatetimeIndex(np.unique(m1.index.normalize())); days = days[days.dayofweek < 5]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    a_bps = (atr(d1, 14) / d1.close * 1e4).shift(1).reindex(days)
    T = lambda h, m=0: days + pd.Timedelta(hours=h, minutes=m)  # noqa: E731
    px = lambda t: pd.Series(c.asof(t).to_numpy(), index=days)  # noqa: E731
    r = lambda a, b: np.log(px(b) / px(a)) * 1e4  # noqa: E731
    prev_close = px(T(23)).shift(1)
    on = np.log(px(T(16, 30)) / prev_close) * 1e4
    cash = r(T(16, 30), T(23))
    rows = []
    add = lambda lab_, v: rows.append(stat(sym, lab_, v))  # noqa: E731
    add("on->cash (cont+)", np.sign(on) * cash)
    add("eu->cash (cont+)", np.sign(r(T(10), T(16, 30))) * cash)
    add("or30->rest (cont+)", np.sign(r(T(16, 30), T(17))) * r(T(17), T(23)))
    add("h1->rest (cont+)", np.sign(r(T(16, 30), T(17, 30))) * r(T(17, 30), T(23)))
    add("day->last (cont+)", np.sign(r(T(1, 5), T(22))) * r(T(22), T(23)))
    add("asia->eu (cont+)", np.sign(r(T(1, 5), T(10))) * r(T(10), T(16, 30)))
    add("london open 10:00->10:30 -> 10:30-16:30 (cont+)", np.sign(r(T(10), T(10, 30))) * r(T(10, 30), T(16, 30)))
    g = on / a_bps
    for lo, hi in ((-9, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 9)):
        mk = (g > lo) & (g <= hi)
        add(f"gap {lo:+.1f}..{hi:+.1f} ATR -> first hour (long)", r(T(16, 30), T(17, 30))[mk])
        add(f"gap {lo:+.1f}..{hi:+.1f} ATR -> cash session (long)", cash[mk])
    big = cash.abs() / a_bps
    add("big cash day (>0.7 ATR) -> next Asia 01:05-10:00 (cont+)", (np.sign(cash) * r(T(1, 5), T(10)).shift(-1))[big > 0.7])
    add("cash session -> next day 01:05->16:30 (cont+)", np.sign(cash) * r(T(1, 5), T(16, 30)).shift(-1))
    return [x for x in rows if x]


if __name__ == "__main__":
    rows = []
    for s in ("NAS100", "DJ30", "GER40", "XAUUSD"):
        rows += run(s)
    T = pd.DataFrame(rows)
    (lab.REPORTS / "EXP-091").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-091" / "intraday_patterns.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"tests {len(T)}")
    print(T.sort_values("t", key=abs, ascending=False).head(30).to_string(index=False))

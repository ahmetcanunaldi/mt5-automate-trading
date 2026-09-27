"""EXP-075: physical-demand and futures-flow calendar on D1 2008-2026 (close-to-close, excess over the average
daily drift). Event-level statistics (one window return per event), era consistency, Bonferroni reading.

  CNY        : Chinese New Year (SGE closed a week; pre-holiday physical buying)
  DHANTERAS  : Indian gold-buying festival (2 days before Diwali)
  COMEX_FND  : first notice day for the active gold months (last business day of Jan/Mar/May/Jul/Sep/Nov) —
               funds roll long futures before it. Control: last business day of the other months.
  QEND       : last trading day of a quarter; YEND last trading day of the year
  THANKS     : US Thanksgiving week; XMAS: Dec 20 -> year end
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

CNY = ["2008-02-07", "2009-01-26", "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
       "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22",
       "2024-02-10", "2025-01-29", "2026-02-17"]
DIWALI = ["2008-10-28", "2009-10-17", "2010-11-05", "2011-10-26", "2012-11-13", "2013-11-03", "2014-10-23",
          "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14", "2021-11-04",
          "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"]
ERAS = [(2008, 2012), (2013, 2018), (2019, 2022), (2023, 2026)]


def last_bdays(idx, months):
    s = pd.Series(idx, index=idx)
    last = s.groupby(idx.to_period("M")).max()
    return pd.DatetimeIndex([t for t in last if t.month in months])


def thanksgiving(y):
    nov = pd.Timestamp(year=y, month=11, day=1)
    return nov + pd.Timedelta(days=(3 - nov.dayofweek) % 7 + 21)


def window_stats(name, r, pos, events, a, b, drift):
    vals, yrs = [], []
    for ev in events:
        e = pos.searchsorted(ev)                       # first trading day >= event
        if e + a < 1 or e + b >= len(pos):
            continue
        w = r.iloc[e + a: e + b + 1]
        vals.append((w - drift).sum()); yrs.append(pos[e].year)
    v = pd.Series(vals, index=yrs)
    er = [round(v[(v.index >= p) & (v.index <= q)].mean(), 1) for p, q in ERAS]
    return {"event": name, "win": f"[{a:+d},{b:+d}]", "n": len(v), "excess_bps": round(v.mean(), 1),
            "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2) if v.std() > 0 else 0, "hit": round((v > 0).mean(), 2),
            "eras": er, "eras_same": sum(np.sign(e) == np.sign(v.mean()) for e in er if not np.isnan(e))}


if __name__ == "__main__":
    d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
    d = d[d.index.dayofweek < 5]
    r = np.log(d.close).diff().dropna().loc["2008":] * 1e4
    pos = r.index
    drift = r.mean()
    ev = {
        "CNY": pd.DatetimeIndex(pd.to_datetime(CNY)),
        "DHANTERAS": pd.DatetimeIndex(pd.to_datetime(DIWALI)) - pd.Timedelta(days=2),
        "COMEX_FND": last_bdays(pos, {1, 3, 5, 7, 9, 11}),
        "EVEN_MEND": last_bdays(pos, {2, 4, 6, 8, 10, 12}),
        "QEND": last_bdays(pos, {3, 6, 9, 12}),
        "THANKS": pd.DatetimeIndex([thanksgiving(y) for y in range(2008, 2026)]),
        "XMAS": pd.DatetimeIndex([pd.Timestamp(f"{y}-12-20") for y in range(2008, 2026)]),
    }
    rows = []
    for name, wins in (("CNY", [(-10, -1), (-5, -1), (0, 4), (5, 14)]), ("DHANTERAS", [(-10, -1), (-5, -1), (0, 4)]),
                       ("COMEX_FND", [(-8, -4), (-3, -1), (0, 0), (1, 3)]), ("EVEN_MEND", [(-8, -4), (-3, -1), (0, 0), (1, 3)]),
                       ("QEND", [(-3, -1), (0, 0), (1, 1)]), ("THANKS", [(-3, -1), (0, 2)]), ("XMAS", [(0, 7)])):
        for a, b in wins:
            rows.append(window_stats(name, r, pos, ev[name], a, b, drift))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"avg daily drift {drift:.1f} bps; {len(T)} tests -> Bonferroni |t| > 3.0")
    print(T.to_string(index=False))
    (lab.REPORTS / "EXP-075").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-075" / "flow_calendar.csv", index=False)

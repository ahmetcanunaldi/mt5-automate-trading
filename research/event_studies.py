"""EXP-030 intraday momentum and EXP-031 pre-news drift (M1 2019-2026, server time).

EXP-030: predictor window return vs later target window return, same day. Report corr, t of sign-strategy
         (target return signed by predictor sign, in bps), per-year signs.
EXP-031: returns in the hours BEFORE US high-impact releases (FOMC, NFP, CPI...), windows ending 10 minutes
         before the release (our flatten rule), by event type; per-year signs.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, lab  # noqa: E402

m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2019":]
c = m1["close"]


def at(days, minute):
    t = days + pd.Timedelta(minutes=minute)
    return pd.Series(c.asof(t).to_numpy() if len(t) else [], index=days)


def momentum():
    days = pd.DatetimeIndex(np.unique(m1.index.normalize()))
    days = days[days.dayofweek < 5]
    pts = {m: at(days, m) for m in (65, 600, 930, 990, 1020, 1200, 1320, 1410)}
    prev_close = pts[1410].shift(1)
    wins = {"prevclose->10:00": (prev_close, pts[600]), "01:05->16:30": (pts[65], pts[990]),
            "NY 16:30->17:00": (pts[990], pts[1020]), "16:30->20:00": (pts[990], pts[1200]),
            "01:05->20:00": (pts[65], pts[1200])}
    tgts = {"20:00->23:30": (pts[1200], pts[1410]), "22:00->23:30": (pts[1320], pts[1410]),
            "17:00->23:30": (pts[1020], pts[1410])}
    print("=== EXP-030 intraday momentum (bps) ===")
    for pn, (a, b) in wins.items():
        pr = np.log(b / a)
        for tn, (x, y) in tgts.items():
            if pn.split("->")[-1] > tn.split("->")[0] and not pn.startswith("prev"):
                continue
            tr = np.log(y / x) * 1e4
            v = (np.sign(pr) * tr).dropna()
            yr = v.groupby(v.index.year).mean()
            print(f"{pn:<17} -> {tn:<13} corr {pr.corr(tr):+.3f}  signed {v.mean():+.2f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}"
                  f"  yrs+ {int((yr > 0).sum())}/{len(yr)}")


def prenews():
    cal = pd.read_csv(lab.DATA / "calendar_usd_high.csv")
    t_utc = pd.DatetimeIndex(pd.to_datetime(cal.time_server) - pd.Timedelta(hours=3))
    cal["t"] = calendar_news.utc_to_server(t_utc)
    cal = cal[(cal.t >= "2019-01-05") & (cal.t <= "2026-09-20")]
    groups = {"FOMC": ["fed-interest-rate-decision"], "NFP": ["nonfarm-payrolls"],
              "CPI": ["consumer-price-index-mm", "consumer-price-index-yy", "consumer-price-index-ex-food-energy-mm"],
              "ALL": list(cal.event_code.unique())}
    print("\n=== EXP-031 pre-news drift: return from T-h to T-10min (bps) ===")
    for gname, codes in groups.items():
        ev = pd.DatetimeIndex(sorted(set(cal[cal.event_code.isin(codes)].t)))
        for h in (1, 3, 6, 24):
            a = c.asof(ev - pd.Timedelta(hours=h)); b = c.asof(ev - pd.Timedelta(minutes=10))
            r = pd.Series(np.log(b.to_numpy() / a.to_numpy()) * 1e4, index=ev).dropna()
            yr = r.groupby(r.index.year).mean()
            print(f"{gname:<5} {h:>2}h before: n {len(r):>4} mean {r.mean():+6.2f} bps t {r.mean() / r.std() * np.sqrt(len(r)):+.2f}"
                  f"  yrs+ {int((yr > 0).sum())}/{len(yr)}")


if __name__ == "__main__":
    momentum()
    prenews()

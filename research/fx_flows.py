"""EXP-079: symbol-specific structure for XAGUSD / EURUSD / USDJPY, M1 2019-2026.

(a) Hour-of-day map (server hour of entry, all weekdays pooled and per weekday) x hold 1/2/4/8 h: bps, t,
    halves agreement, years with the same sign, and the round-trip cost in bps (median bar spread + $7/lot +
    2 x slippage) -> "net" = |bps| - cost.
(b) USDJPY gotobi anomaly: on gotobi days (5, 10, 15, 20, 25, month end; weekend -> previous Friday) USD demand
    into the Tokyo fix 09:55 JST. Windows in JST (UTC+9): 07:00->09:55, 09:55->12:00; compared with other days.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, lab, symbols  # noqa: E402


def cost_bps(sym, m1):
    sp = symbols.SPECS[sym]; c = symbols.COSTS[sym]
    px = m1.close.median()
    spread = max(m1.spread.median(), c.min_spread_pts) * sp["point"]
    comm = c.commission_per_lot / (sp["contract"] if sp["quote"] == "USD" else sp["contract"] / px)
    return (spread + comm + 2 * c.slippage_pts * sp["point"]) / px * 1e4


def hour_map(sym, m1):
    h1 = m1["close"].resample("1h").last().dropna()
    op = m1["open"].resample("1h").first().reindex(h1.index)
    cb = cost_bps(sym, m1)
    rows = []
    for hold in (1, 2, 4, 8):
        fwd = np.log(h1.shift(-(hold - 1)) / op) * 1e4
        same = h1.index.normalize() == pd.Series(h1.index, index=h1.index).shift(-(hold - 1)).dt.normalize()
        f = pd.DataFrame({"r": fwd, "ok": same.to_numpy()}, index=h1.index)
        f = f[f.ok & f.r.notna() & (f.index.dayofweek < 5)]
        f["dow"] = f.index.dayofweek; f["hr"] = f.index.hour; f["yr"] = f.index.year
        for key, g in list(f.groupby("hr")) + list(f.groupby(["dow", "hr"])):
            if len(g) < 150:
                continue
            m = g.r.mean(); t = m / g.r.std() * np.sqrt(len(g))
            ym = g.groupby("yr").r.mean(); a, b = g[g.yr <= 2022].r.mean(), g[g.yr >= 2023].r.mean()
            dw, hr = (key if isinstance(key, tuple) else ("all", key))
            rows.append({"sym": sym, "dow": dw, "hour": hr, "hold_h": hold, "n": len(g), "bps": round(m, 2),
                         "t": round(t, 2), "net_bps": round(abs(m) - cb, 2), "yrs_same": round((np.sign(ym) == np.sign(m)).mean(), 2),
                         "halves_agree": bool(np.sign(a) == np.sign(b))})
    return pd.DataFrame(rows), cb


def gotobi_days(days):
    out = set()
    by_month = pd.Series(days, index=days).groupby(days.to_period("M"))
    for per, s in by_month:
        idx = pd.DatetimeIndex(s.values)
        for dom in (5, 10, 15, 20, 25, 31):
            target = pd.Timestamp(year=per.year, month=per.month, day=min(dom, per.days_in_month))
            prev = idx[idx <= target]
            if len(prev):
                out.add(prev[-1])
    return pd.DatetimeIndex(sorted(out))


def jst_to_server(dates, hhmm):
    loc = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {hhmm}") for d in dates]) - pd.Timedelta(hours=9)
    return calendar_news.utc_to_server(loc)


def gotobi(m1):
    c = m1.close
    days = pd.DatetimeIndex(np.unique(m1.index.normalize())); days = days[days.dayofweek < 5]
    gd = gotobi_days(days)
    rows = []
    for a, b in (("07:00", "09:55"), ("08:00", "09:55"), ("09:55", "12:00"), ("09:55", "15:00")):
        ta, tb = jst_to_server(days, a), jst_to_server(days, b)
        r = pd.Series(np.log(c.asof(tb).to_numpy() / c.asof(ta).to_numpy()) * 1e4, index=days)
        for lab_, mask in (("gotobi", days.isin(gd)), ("other", ~days.isin(gd))):
            v = r[mask].dropna(); yr = v.groupby(v.index.year).mean()
            rows.append({"window_JST": f"{a}->{b}", "days": lab_, "n": len(v), "bps": round(v.mean(), 2),
                         "t": round(v.mean() / v.std() * np.sqrt(len(v)), 2), "hit": round((v > 0).mean(), 2),
                         "yrs_pos": int((yr > 0).sum()), "by_year": yr.round(1).to_dict()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    out = []
    for sym in ("XAGUSD", "EURUSD", "USDJPY"):
        m1 = symbols.load_m1(sym).loc["2019":"2026-09-25"]
        T, cb = hour_map(sym, m1)
        out.append(T)
        good = T[(T.t.abs() > 3) & T.halves_agree & (T.yrs_same >= 0.75)].sort_values("t", key=abs, ascending=False)
        print(f"\n=== {sym}: round-trip cost {cb:.2f} bps; tests {len(T)}; |t|>3 & stable: {len(good)}; of which net > 0: {(good.net_bps > 0).sum()}")
        print(good.head(15).to_string(index=False))
        if sym == "USDJPY":
            print("\n--- gotobi ---\n", gotobi(m1).to_string(index=False))
    (lab.REPORTS / "EXP-079").mkdir(exist_ok=True)
    pd.concat(out).to_csv(lab.REPORTS / "EXP-079" / "hour_maps.csv", index=False)

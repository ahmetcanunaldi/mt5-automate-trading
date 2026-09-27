"""FXR-002: economically motivated FX effects on the 7-currency panel (DEV 2015-2024), each pre-specified from the
literature (no parameter search):
  A. Carry (Lustig-Roussanov-Verdelhan; Koijen et al.): currency weight = cross-sectional demeaned 3-month
     interbank rate (FRED, monthly, 2-month publication lag). Daily at server hour H, weekend flat (Friday close,
     Monday re-entry), return = spot + interest accrual (Wednesday triple, no weekend nights) - broker swap markup
     (0.5 %/yr per side) - spread/commission/slippage on every change.
  B. Session effect (Breedon & Ranaldo 2013, "Intraday patterns in FX returns and order flow"): currencies
     depreciate during their home trading hours. EUR/GBP/CHF: short in the European session, long in the US session;
     JPY/AUD: short in the Asian session, long in the European session. Server-time sessions: Asia 01-09,
     Europe 09-15, US 15-23.
  C. Month-end hedge rebalancing (Melvin & Prins 2015): last trading day of the month, server 13:00 -> 18:00
     (London 4 pm fix); prediction: USD falls at the fix when US equities outperformed foreign equities
     month-to-date (DJ30 vs GER40), and vice versa."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.fx import data as F  # noqa: E402
from research.fx.fxr001_factors import daily_grid  # noqa: E402

OUT = lab.REPORTS / "fx" / "FXR-002"
MARKUP = 0.5          # % per year per side of the swap vs the interest differential


def shp(x, per=252):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / x.std() * np.sqrt(per)), 2) if x.std() > 0 else 0.0


def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / x.std() * np.sqrt(len(x))), 2) if len(x) > 2 else np.nan


def costs_table():
    sp = F.spread_bp()
    return pd.DataFrame({F.CCY[p]: [F.cost_bp(p, y, sp) for y in sp.index] for p in F.PAIRS}, index=sp.index)


def rates():
    R = pd.concat({c: pd.read_csv(lab.DATA / "fred" / f"{c}.csv", index_col=0, parse_dates=True).iloc[:, 0]
                   for c in ["USD"] + [F.CCY[p] for p in F.PAIRS]}, axis=1).apply(pd.to_numeric, errors="coerce")
    R.index = R.index + pd.DateOffset(months=2)             # publication lag (conservative)
    return R.ffill()


def carry(H=9):
    at, fwd, _ = daily_grid(H)
    R = rates().reindex(at.index, method="ffill")[at.columns]
    D = R.sub(R.mean(1), axis=0)
    W = D.drop(columns="USD")
    W = W.div(W.abs().sum(1), axis=0)                          # pair gross = 1
    dow = at.index.dayofweek
    nights = np.select([dow == 2, dow == 4], [3, 0], 1)          # Wed triple, Friday closes before the weekend
    diff = R.drop(columns="USD").sub(R["USD"], axis=0) / 100 / 360   # daily differential vs USD
    accr = (W * diff).sum(1) * nights * 1e4 - W.abs().sum(1) * MARKUP / 100 / 360 * nights * 1e4
    spot = (W * fwd.drop(columns="USD")).sum(1) * 1e4
    C = costs_table()
    cyear = C.reindex(at.index.year).set_axis(at.index)
    prev = W.shift(1).mul(np.where(dow == 0, 0.0, 1.0), axis=0)  # Monday: re-open from flat
    turn = (W - prev.fillna(0)).abs()
    fri = pd.Series(dow == 4, index=at.index)
    cost = (turn * cyear).sum(1) / 2 + (W.abs() * cyear).sum(1).where(fri, 0) / 2
    # entry/exit legs: each side half the round-turn cost; Friday close is an exit of the whole book
    net = (spot + accr - cost).loc[fwd.index].dropna()
    parts = pd.DataFrame({"spot": spot, "carry": accr, "cost": -cost}).loc[net.index]
    yr = net.groupby(net.index.year).apply(lambda x: shp(x))
    return {"SR_net": shp(net), "SR_spot_only": shp(parts.spot), "bp_day_spot": round(parts.spot.mean(), 2),
            "bp_day_carry": round(parts.carry.mean(), 2), "bp_day_cost": round(parts.cost.mean(), 2),
            "SR_2015_19": shp(net[net.index.year <= 2019]), "SR_2020_24": shp(net[net.index.year >= 2020]),
            "by_year": yr.to_dict()}, net


def sessions():
    L = F.ccy_log("H1", end=F.DEV_END)
    L = L[L.index >= "2015-01-01"]
    C = costs_table()
    blocks = {"asia": (1, 9), "europe": (9, 15), "us": (15, 23)}
    rows, pnl = [], []
    day = L.index.normalize()
    for name, (a, b) in blocks.items():
        pa = L[L.index.hour == a]; pb = L[L.index.hour == b]
        pa.index = pa.index.normalize(); pb.index = pb.index.normalize()
        pa = pa[~pa.index.duplicated()]; pb = pb[~pb.index.duplicated()]
        r = (pb - pa).dropna() * 1e4
        for c in r.columns[1:]:
            rows.append({"session": name, "ccy": c, "mean_bp": round(r[c].mean(), 2), "t": tstat(r[c]),
                         "t_2015_19": tstat(r[c][r.index.year <= 2019]), "t_2020_24": tstat(r[c][r.index.year >= 2020])})
        pnl.append((name, r))
    T = pd.DataFrame(rows)
    # pre-specified rule
    rule = {("EUR", "europe"): -1, ("GBP", "europe"): -1, ("CHF", "europe"): -1, ("EUR", "us"): 1, ("GBP", "us"): 1,
            ("CHF", "us"): 1, ("JPY", "asia"): -1, ("AUD", "asia"): -1, ("JPY", "europe"): 1, ("AUD", "europe"): 1}
    trades = []
    for (c, s), sg in rule.items():
        r = dict(pnl)[s][c] * sg
        cost = C[c].reindex(r.index.year).to_numpy()
        trades.append(pd.DataFrame({"g": r.to_numpy(), "n": r.to_numpy() - cost}, index=r.index).assign(ccy=c, sess=s))
    Tr = pd.concat(trades)
    daily = Tr.groupby(level=0)[["g", "n"]].sum()
    per = Tr.groupby(["ccy", "sess"]).agg(mean_gross_bp=("g", "mean"), mean_net_bp=("n", "mean"),
                                          t=("g", lambda x: tstat(x))).round(2)
    return T, per, {"SR_gross": shp(daily.g), "SR_net": shp(daily.n), "gross_bp_trade": round(Tr.g.mean(), 2),
                    "net_bp_trade": round(Tr.n.mean(), 2)}


def month_end():
    L = F.ccy_log("H1", end=F.DEV_END)
    L = L[L.index >= "2013-01-01"]
    usd = -L.drop(columns="USD").mean(1)                           # USD vs the basket
    eq = {s: pd.read_parquet(lab.DATA / f"{s}_D1_long.parquet").close for s in ("DJ30", "GER40")}
    days = pd.Series(L.index.normalize().unique())
    last = days.groupby(days.dt.to_period("M")).max()
    rows = []
    for d in last:
        a, b = d + pd.Timedelta(hours=13), d + pd.Timedelta(hours=18)
        if a not in L.index or b not in L.index:
            continue
        m0 = d.to_period("M").start_time
        mtd = {}
        for s, px in eq.items():
            p = px.loc[:d - pd.Timedelta(days=1)]
            base = px.loc[:m0 - pd.Timedelta(days=1)]
            if len(p) and len(base):
                mtd[s] = np.log(p.iloc[-1] / base.iloc[-1])
        if len(mtd) < 2:
            continue
        rows.append({"date": d, "usd_bp": (usd[b] - usd[a]) * 1e4, "eur_bp": (L.EUR[b] - L.EUR[a]) * 1e4,
                     "us_minus_eu": mtd["DJ30"] - mtd["GER40"]})
    M = pd.DataFrame(rows).set_index("date")
    sig = -np.sign(M.us_minus_eu)                                   # predicted USD direction
    pnl = sig * M.usd_bp
    pnl_eur = np.sign(M.us_minus_eu) * M.eur_bp                    # US outperformed -> EUR up vs USD
    dev = M.index.year <= 2024
    corr = np.corrcoef(M.us_minus_eu, M.usd_bp)[0, 1]
    return M, {"months": len(M), "corr(US-EU mtd, USD at fix)": round(corr, 3), "usd_mean_bp": round(M.usd_bp.mean(), 2),
               "usd_t": tstat(M.usd_bp), "rule_bp": round(pnl.mean(), 2), "rule_t": tstat(pnl),
               "rule_hit": round((pnl > 0).mean(), 3), "eur_rule_bp": round(pnl_eur.mean(), 2), "eur_rule_t": tstat(pnl_eur),
               "rule_t_2013_18": tstat(pnl[M.index.year <= 2018]), "rule_t_2019_24": tstat(pnl[(M.index.year >= 2019) & dev])}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("== A. carry")
    for H in (9, 16):
        res, net = carry(H)
        print("H", H, res)
        net.to_csv(OUT / f"carry_H{H}.csv")
    print("== B. sessions")
    T, per, s = sessions()
    T.to_csv(OUT / "sessions.csv", index=False)
    print(T.pivot(index="ccy", columns="session", values="t").to_string())
    print(per.to_string()); print(s)
    print("== C. month-end fix")
    M, r = month_end()
    M.to_csv(OUT / "month_end.csv"); print(r)

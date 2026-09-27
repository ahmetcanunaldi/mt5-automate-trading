"""FXR-001: factor library scan on the 8-currency panel (DEV only: data start .. 2024-12-31).

Daily grid: currency log values sampled at server hour H on every weekday; forward return = to the same hour of the
next weekday, except Fridays -> Friday 23:00 (weekend flat). Every signal is a currency score s[t, c] built from
information up to t. Reported per signal:
  * XS IC: daily Spearman corr of the cross-sectionally demeaned score with the next-day currency return (8 ccys)
  * TS IC: pooled corr of score with next-day return per currency (vs USD)
  * XS portfolio: weights = demeaned score / sum |.|, gross 1 -> daily return, turnover, cost (per-currency major cost)
    Sharpe gross / net, split data start-2019 (discovery) vs 2020-2024 (validation), sign consistency by year.
Every configuration is appended to the trial ledger for deflated-Sharpe accounting."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from research import lab  # noqa: E402
from research.fx import data as F  # noqa: E402

OUT = lab.REPORTS / "fx" / "FXR-001"
H = int(sys.argv[1]) if len(sys.argv) > 1 else 9


def daily_grid(H=H, end=F.DEV_END):
    L = F.ccy_log("H1", end=end)
    at = L[L.index.hour == H]
    at = at[~at.index.normalize().duplicated()]
    fri_close = L[(L.index.dayofweek == 4)].groupby(L[(L.index.dayofweek == 4)].index.normalize()).last()
    nxt = at.shift(-1)
    fwd = nxt - at
    fri = at.index.dayofweek == 4
    fc = fri_close.reindex(at.index.normalize()).to_numpy()
    fwd[fri] = fc[fri] - at[fri].to_numpy()
    gap = (pd.Series(at.index, index=at.index).shift(-1) - at.index).dt.days
    fwd[(~fri) & (gap.to_numpy() > 1)] = np.nan          # holidays / missing days
    daily_ret = at.diff()                                  # close-to-close at H (includes weekend for Mondays)
    return at, fwd.iloc[:-1], daily_ret


def xs_demean(S):
    return S.sub(S.mean(1), axis=0)


def signals(at, dret, Hr):
    vol = dret.iloc[:, 1:].rolling(60).std()
    vol.insert(0, "USD", vol.mean(1))
    z = lambda x: x / vol                                  # noqa: E731
    S = {}
    for k in (1, 2, 5, 10, 20, 60, 120, 250):
        S[f"mom{k}"] = z((at - at.shift(k)) / np.sqrt(k))
    S["mom_12_1"] = z((at.shift(21) - at.shift(252)) / np.sqrt(231))
    # distance from 20/60-day moving average (trend strength)
    for k in (20, 60):
        S[f"ma_gap{k}"] = z(at - at.rolling(k).mean())
    # residual reversal: remove the first principal component (dollar / risk factor) of 60-day XS returns
    R = dret.fillna(0)
    res = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    X = R.to_numpy()
    for i in range(120, len(R)):
        W = X[i - 120:i]; W = W - W.mean(0)
        u = np.linalg.svd(W, full_matrices=False)[2][0]
        x = X[i] - X[i].mean()
        res.iloc[i] = x - (x @ u) * u
    for k in (1, 3, 5):
        S[f"resid{k}"] = z(res.rolling(k).sum())
    # volatility / range based
    S["lowvol"] = -vol.rank(axis=1)
    S["vol_change"] = -(dret.iloc[:, 1:].rolling(5).std() / vol.iloc[:, 1:]).reindex(columns=at.columns).fillna(0)
    # intraday: last session return (previous 8 hours to H) and overnight
    Hl = Hr
    for h in (4, 8):
        prev = Hl.shift(h).reindex(at.index)
        S[f"last{h}h"] = z(at - prev)
    # calendar (pooled currency x weekday / turn-of-month are left to FXR-002: they need per-currency parameters)
    return S


def evaluate(name, S, fwd, costs):
    S = S.reindex(fwd.index)
    D = xs_demean(S)
    ok = D.notna().all(1) & fwd.notna().all(1)
    D, Y = D[ok], fwd[ok]
    ic = np.array([stats.spearmanr(a, b)[0] for a, b in zip(D.to_numpy(), Y.to_numpy())])
    ts = S[ok].iloc[:, 1:].to_numpy().ravel(); yy = Y.iloc[:, 1:].to_numpy().ravel()
    m = np.isfinite(ts) & np.isfinite(yy)
    ts_ic = np.corrcoef(ts[m], yy[m])[0, 1]
    W = D.div(D.abs().sum(1), axis=0)
    gross = (W * Y).sum(1) * 1e4                            # bp per day on gross 1
    turn = W.diff().abs().iloc[:, 1:].fillna(0)             # USD leg is implicit in the majors
    cost = (turn * costs.reindex(W.index.year).set_axis(W.index)).sum(1)
    net = gross - cost
    yrs = W.index.year
    shp = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0  # noqa: E731
    by_year = pd.Series(ic).groupby(yrs).mean()
    a, b = yrs <= 2019, yrs >= 2020
    return {"signal": name, "days": int(ok.sum()), "xs_ic": round(ic.mean(), 4), "xs_ic_t": round(ic.mean() / ic.std() * np.sqrt(len(ic)), 2),
            "ts_ic": round(ts_ic, 4), "ic_pos_years": f"{(by_year > 0).sum()}/{len(by_year)}",
            "gross_bp_day": round(gross.mean(), 2), "cost_bp_day": round(cost.mean(), 2), "turnover": round(turn.sum(1).mean(), 2),
            "SR_gross": round(shp(gross), 2), "SR_net": round(shp(net), 2),
            "SR_net_disc": round(shp(net[a]), 2), "SR_net_val": round(shp(net[b]), 2),
            "SR_gross_disc": round(shp(gross[a]), 2), "SR_gross_val": round(shp(gross[b]), 2)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    at, fwd, dret = daily_grid()
    Hr = F.ccy_log("H1", end=F.DEV_END)
    sp = F.spread_bp()
    costs = pd.DataFrame({F.CCY[p]: [F.cost_bp(p, y, sp) for y in sp.index] for p in F.PAIRS}, index=sp.index)
    print("cost bp (round turn) by year:\n", costs.round(2).to_string())
    print("panel", at.index[0], "->", at.index[-1], len(at), "days")
    S = signals(at, dret, Hr)
    rows = [evaluate(k, v, fwd, costs) for k, v in S.items()]
    T = pd.DataFrame(rows).sort_values("xs_ic_t", key=abs, ascending=False)
    T.to_csv(OUT / f"factors_H{H}.csv", index=False)
    print(T.to_string(index=False))
    with open(lab.REPORTS / "quant" / "trials.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"exp": "FXR-001", "H": H, **r}) + "\n")

"""Volatility models and forecast evaluation.

har_design / har_fit / har_forecast : HAR-RV (Corsi 2009) on log realized variance, optional leverage term
                                      (negative daily return) and jump component (RV - BV)
garch_forecasts                     : GARCH(1,1) / GJR-GARCH(1,1,1) with Student-t errors (arch), one-step variance
rough_hurst                         : scaling of E|log sigma_{t+D} - log sigma_t|^q ~ D^{qH} (Gatheral, Jaisson &
                                      Rosenbaum 2018) -> H of the log-volatility process
qlike / mz_r2                       : QLIKE loss (robust to noisy proxies, Patton 2011) and Mincer-Zarnowitz R^2
walk_forward_rv                     : yearly expanding-window one-day-ahead RV forecasts for several models
"""
import numpy as np
import pandas as pd
from arch import arch_model


def har_design(rv: pd.Series, ret: pd.Series | None = None, jump: pd.Series | None = None):
    lr = np.log(rv.clip(lower=rv[rv > 0].min()))
    X = pd.DataFrame({"d": lr, "w": lr.rolling(5).mean(), "m": lr.rolling(22).mean()})
    if ret is not None:
        X["lev"] = np.minimum(ret, 0) / np.sqrt(rv)                # standardized negative return
    if jump is not None:
        X["jump"] = jump
    y = lr.shift(-1).rename("y")
    return X, y


def har_fit(X, y):
    m = X.notna().all(axis=1) & y.notna()
    A = np.column_stack([np.ones(m.sum()), X[m].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, y[m].to_numpy(), rcond=None)
    resid = y[m].to_numpy() - A @ beta
    return beta, float(resid.var())


def har_forecast(X, beta, s2):
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    return pd.Series(np.exp(A @ beta + s2 / 2), index=X.index)     # log-normal bias correction


def garch_forecasts(ret: pd.Series, train_end, kind="gjr"):
    """Fit on returns up to train_end (inclusive), then filter forward with fixed parameters: one-step-ahead
    conditional variance for every later day. Returns in % units internally (arch convention)."""
    r = ret.dropna() * 100
    o = 1 if kind == "gjr" else 0
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t")
    res = am.fit(last_obs=r.index[r.index <= train_end][-1], disp="off")
    f = res.forecast(horizon=1, start=r.index[r.index > train_end][0], reindex=False)
    var = f.variance["h.1"] / 1e4                                  # back to squared log return units
    # forecast made at t for t+1 -> align to the target day
    return pd.Series(var.to_numpy(), index=var.index).shift(1).dropna(), res.params


def rough_hurst(rv: pd.Series, qs=(0.5, 1.0, 1.5, 2.0), lags=range(1, 31)):
    ls = 0.5 * np.log(rv[rv > 0])
    out = {}
    for q in qs:
        m = [np.mean(np.abs(ls.diff(d).dropna()) ** q) for d in lags]
        slope = np.polyfit(np.log(list(lags)), np.log(m), 1)[0]
        out[q] = slope / q
    return float(np.mean(list(out.values()))), out


def qlike(proxy, fcst):
    p, f = np.asarray(proxy, float), np.asarray(fcst, float)
    m = np.isfinite(p) & np.isfinite(f) & (p > 0) & (f > 0)
    return float(np.mean(p[m] / f[m] - np.log(p[m] / f[m]) - 1))


def mz_r2(proxy, fcst):
    p, f = np.log(np.asarray(proxy, float)), np.log(np.asarray(fcst, float))
    m = np.isfinite(p) & np.isfinite(f)
    return float(np.corrcoef(p[m], f[m])[0, 1] ** 2)


def walk_forward_rv(R: pd.DataFrame, first_year, last_year):
    """R: output of data.realized (rv, bv, jump, ret). One-day-ahead forecasts of rv for each test year with models
    fitted on all earlier days. Returns DataFrame [rv (target), naive, ewma, har, har_lev_j, garch, gjr]."""
    out = []
    X0, y = har_design(R.rv)
    X1, _ = har_design(R.rv, R.ret, R.jump)
    for Y in range(first_year, last_year + 1):
        tr = R.index < pd.Timestamp(f"{Y}-01-01")
        te = (R.index >= pd.Timestamp(f"{Y}-01-01")) & (R.index < pd.Timestamp(f"{Y + 1}-01-01"))
        if tr.sum() < 250 or te.sum() == 0:
            continue
        b0, s0 = har_fit(X0[tr], y[tr]); b1, s1 = har_fit(X1[tr], y[tr])
        f = pd.DataFrame(index=R.index[te])
        f["rv"] = R.rv[te]
        f["naive"] = R.rv.shift(1)[te]
        f["ewma"] = R.rv.ewm(alpha=0.06).mean().shift(1)[te]
        f["har"] = har_forecast(X0, b0, s0).shift(1)[te]
        f["har_lev_j"] = har_forecast(X1.fillna(0), b1, s1).shift(1)[te]
        daily_ret = np.log(R.close).diff()
        for kind in ("garch", "gjr"):
            g, _ = garch_forecasts(daily_ret, R.index[tr][-1], kind)
            # GARCH forecasts close-to-close variance (incl. overnight); scale to the intraday RV level in train
            scale = (R.rv[tr].mean() / (daily_ret[tr] ** 2).mean())
            f[kind] = (g * scale).reindex(f.index)
        out.append(f)
    return pd.concat(out)

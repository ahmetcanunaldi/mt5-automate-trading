"""Cointegration and dynamic hedge ratios for spread construction (log prices).

engle_granger : OLS y = a + b x, ADF p-value of the residual (statsmodels)
johansen      : trace statistics / first eigenvector (statsmodels coint_johansen)
kalman_hedge  : time-varying (alpha_t, beta_t) random-walk state, y_t = alpha_t + beta_t x_t + e_t; returns the
                one-step-ahead (causal) spread y_t - alpha_{t|t-1} - beta_{t|t-1} x_t and its predicted variance
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger(y, x):
    b, a = np.polyfit(x, y, 1)
    e = y - (a + b * x)
    return {"alpha": float(a), "beta": float(b), "adf_p": float(adfuller(np.asarray(e), maxlag=10, autolag=None)[1]), "resid": e}


def johansen(P: pd.DataFrame, k_ar_diff=1):
    r = coint_johansen(P.to_numpy(), 0, k_ar_diff)
    return {"trace": r.lr1, "crit95": r.cvt[:, 1], "vec": r.evec[:, 0]}


def kalman_hedge(y, x, delta=1e-5, r_var=None):
    """delta: state noise share (Chan's formulation, Vw = delta/(1-delta) I). Returns DataFrame alpha, beta,
    spread (innovation), s2 (innovation variance)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    Vw = delta / (1 - delta) * np.eye(2)
    Ve = r_var if r_var is not None else np.var(np.diff(y)) if n > 2 else 1e-4
    th = np.zeros(2); P = np.eye(2)
    out = np.zeros((n, 4))
    for t in range(n):
        F = np.array([1.0, x[t]])
        R = P + Vw
        yhat = F @ th
        Q = F @ R @ F + Ve
        e = y[t] - yhat
        K = R @ F / Q
        out[t] = (th[0], th[1], e, Q)                    # prediction made BEFORE seeing y_t (causal)
        th = th + K * e
        P = R - np.outer(K, F) @ R
    return pd.DataFrame(out, columns=["alpha", "beta", "spread", "s2"])

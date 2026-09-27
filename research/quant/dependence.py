"""Model-free dependence measures for return series.

variance_ratio : Lo-MacKinlay (1988) overlapping VR(q) with the heteroskedasticity-robust z* statistic
autocorr       : lag-k autocorrelation with its i.i.d. standard error and a robust (Lo-MacKinlay delta) SE
dfa_hurst      : detrended fluctuation analysis exponent (0.5 = no memory, > 0.5 persistent, < 0.5 anti-persistent)
mutual_info    : binned mutual information between x and y (quantile bins), in nats
transfer_entropy: TE X->Y with quantile bins (Schreiber 2000), lag 1
lead_lag       : cross-correlation corr(x_t, y_{t+k}) for k in -K..K on synchronous bars
"""
import numpy as np
import pandas as pd


def variance_ratio(r, q):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    n = len(r)
    mu = r.mean()
    e = r - mu
    s1 = (e ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")                         # overlapping q-sums
    m = q * (n - q + 1) * (1 - q / n)
    sq = ((rq - q * mu) ** 2).sum() / m
    vr = sq / s1
    # robust variance of VR (Lo-MacKinlay theta*)
    e2 = e ** 2
    den = (e2.sum()) ** 2
    theta = 0.0
    for j in range(1, q):
        dj = n * (e2[j:] * e2[:-j]).sum() / den
        theta += (2 * (q - j) / q) ** 2 * dj
    z = (vr - 1) / np.sqrt(theta / n) if theta > 0 else 0.0
    return float(vr), float(z)


def autocorr(r, k=1):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    e = r - r.mean()
    n = len(e)
    rho = (e[k:] * e[:-k]).sum() / (e ** 2).sum()
    # robust SE under heteroskedasticity: sqrt(delta_k)/sqrt(n)
    delta = n * ((e[k:] ** 2) * (e[:-k] ** 2)).sum() / ((e ** 2).sum()) ** 2
    return float(rho), float(rho / np.sqrt(delta / n))


def dfa_hurst(r, scales=None):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    y = np.cumsum(r - r.mean())
    n = len(y)
    scales = scales or np.unique(np.logspace(np.log10(16), np.log10(max(32, n // 8)), 14).astype(int))
    F = []
    for s in scales:
        m = n // s
        if m < 4:
            continue
        seg = y[: m * s].reshape(m, s)
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        resid = seg - (A @ coef).T
        F.append((s, np.sqrt((resid ** 2).mean())))
    F = np.array(F)
    return float(np.polyfit(np.log(F[:, 0]), np.log(F[:, 1]), 1)[0])


def _bins(x, k):
    q = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(q, x)


def mutual_info(x, y, k=5):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    bx, by = _bins(x, k), _bins(y, k)
    joint = np.zeros((k, k)); np.add.at(joint, (bx, by), 1); joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def transfer_entropy(x, y, k=3):
    """TE_{X->Y} = I(Y_{t+1}; X_t | Y_t) with quantile bins."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    bx, by = _bins(x, k), _bins(y, k)
    y1, y0, x0 = by[1:], by[:-1], bx[:-1]
    P = np.zeros((k, k, k)); np.add.at(P, (y1, y0, x0), 1); P /= P.sum()
    p_y0x0 = P.sum(0); p_y1y0 = P.sum(2); p_y0 = P.sum((0, 2))
    te = 0.0
    for a in range(k):
        for b in range(k):
            for c in range(k):
                if P[a, b, c] > 0:
                    te += P[a, b, c] * np.log(P[a, b, c] * p_y0[b] / (p_y0x0[b, c] * p_y1y0[a, b]))
    return float(te)


def lead_lag(x: pd.Series, y: pd.Series, K=5):
    """corr(x_t, y_{t+k}); k > 0: x leads y."""
    df = pd.concat([x, y], axis=1, join="inner").dropna()
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    return {k: float(a.corr(b.shift(-k))) for k in range(-K, K + 1)}

"""Jumps and self-excitation.

lee_mykland : jump statistic L_t = r_t / sigma_hat_t with sigma_hat from the bipower variation of the previous K
              intraday returns (Lee & Mykland 2008); critical value from the Gumbel limit at level alpha
hawkes_fit  : univariate Hawkes process with exponential kernel lambda(t) = mu + sum alpha * exp(-beta (t - t_i)),
              exact log-likelihood (Ozaki recursion), MLE by L-BFGS-B; branching ratio n = alpha / beta
hawkes_sim  : Ogata thinning simulation (tests)
"""
import math

import numpy as np
from scipy import optimize


def lee_mykland(r, K=78, alpha=0.01, n_total=None):
    r = np.asarray(r, float)
    ar = np.abs(r)
    bp = np.convolve(ar[1:] * ar[:-1], np.ones(K - 2), "valid") / (K - 2)       # mean of |r_j||r_{j-1}| over window
    sig = np.full(len(r), np.nan)
    sig[K:] = np.sqrt(bp[: len(r) - K])                                          # uses returns strictly before t
    L = r / sig
    n = n_total or len(r)
    c = math.sqrt(2 / math.pi)
    Cn = math.sqrt(2 * math.log(n)) / c - (math.log(math.pi) + math.log(math.log(n))) / (2 * c * math.sqrt(2 * math.log(n)))
    Sn = 1 / (c * math.sqrt(2 * math.log(n)))
    crit = -math.log(-math.log(1 - alpha)) * Sn + Cn
    return L, crit


def _nll(p, t, T):
    mu, a, b = np.exp(p)
    if a >= b:                                     # stationarity (branching ratio < 1)
        return 1e12
    A = 0.0; ll = 0.0; prev = None
    for ti in t:
        if prev is not None:
            A = math.exp(-b * (ti - prev)) * (1 + A)
        ll += math.log(mu + a * A)
        prev = ti
    ll -= mu * T + (a / b) * np.sum(1 - np.exp(-b * (T - t)))
    return -ll


def hawkes_fit(t, T):
    t = np.asarray(t, float)
    x0 = np.log([len(t) / T * 0.5, 0.5, 1.0])
    res = optimize.minimize(_nll, x0, args=(t, T), method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-6})
    mu, a, b = np.exp(res.x)
    return {"mu": mu, "alpha": a, "beta": b, "branching": a / b, "nll": res.fun}


def hawkes_sim(mu, a, b, T, seed=0):
    rng = np.random.default_rng(seed)
    t, out, lam_ex = 0.0, [], 0.0
    while t < T:
        lam_bar = mu + lam_ex
        w = rng.exponential(1 / lam_bar)
        lam_ex *= math.exp(-b * w); t += w
        if t >= T:
            break
        if rng.random() * lam_bar <= mu + lam_ex:
            out.append(t); lam_ex += a
    return np.array(out)

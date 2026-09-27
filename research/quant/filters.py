"""Latent-state filters for drift / regime inference (strictly causal: filtering, never smoothing).

hmm_fit_filter : Gaussian HMM (hmmlearn) fitted on training data; forward filter P(state_t | x_1..x_t) run on any
                 later data with the fitted parameters (discrete-time analogue of the Wonham filter). Returns the
                 filtered probabilities and the one-step predictive mean / variance of the observation.
kalman_drift   : local-level model r_t = mu_t + e_t, mu_t = mu_{t-1} + w_t; returns mu_{t|t} and its variance with
                 q/r chosen by maximum likelihood on the training part.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import optimize, stats


def _forward(X, startprob, A, means, covs):
    n, k = len(X), len(startprob)
    B = np.column_stack([stats.multivariate_normal(means[j], covs[j], allow_singular=True).pdf(X) for j in range(k)])
    if B.ndim == 1:
        B = B[:, None]
    alpha = np.zeros((n, k)); a = startprob.copy()
    for t in range(n):
        a = (a @ A if t > 0 else a) * B[t]
        s = a.sum()
        a = a / s if s > 0 else np.full(k, 1.0 / k)
        alpha[t] = a
    return alpha


def hmm_fit_filter(X_train, X_all, n_states=2, seed=0):
    """X_*: (n, d) arrays. Fit on X_train, filter X_all (the caller slices out the test part). States are sorted by
    the mean of the first feature (state 0 = lowest mean)."""
    m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed, tol=1e-4)
    m.fit(X_train)
    order = np.argsort(m.means_[:, 0])
    means, covs = m.means_[order], m.covars_[order]
    A = m.transmat_[np.ix_(order, order)]
    sp = m.startprob_[order]
    alpha = _forward(X_all, sp, A, means, covs)
    pred = alpha @ A                                     # P(state_{t+1} | F_t)
    mu_next = pred @ means[:, 0]
    var_next = pred @ (covs[:, 0, 0] + means[:, 0] ** 2) - mu_next ** 2
    return {"alpha": alpha, "pred": pred, "mu_next": mu_next, "var_next": var_next, "means": means, "A": A}


def _kf(r, q, rv):
    n = len(r); mu = 0.0; P = 1e-2
    out = np.zeros((n, 2)); ll = 0.0
    for t in range(n):
        P = P + q
        S = P + rv
        e = r[t] - mu
        ll += -0.5 * (np.log(2 * np.pi * S) + e * e / S)
        K = P / S
        mu = mu + K * e; P = (1 - K) * P
        out[t] = (mu, P)
    return out, ll


def kalman_drift(r_train, r_all):
    r_train = np.asarray(r_train, float); r_all = np.asarray(r_all, float)
    v = r_train.var()
    f = lambda p: -_kf(r_train, np.exp(p[0]) * v, np.exp(p[1]) * v)[1]  # noqa: E731
    res = optimize.minimize(f, x0=[np.log(1e-4), 0.0], method="Nelder-Mead", options={"maxiter": 400})
    q, rv = np.exp(res.x[0]) * v, np.exp(res.x[1]) * v
    out, _ = _kf(r_all, q, rv)
    return pd.DataFrame({"mu": out[:, 0], "P": out[:, 1]}), {"q": q, "r": rv, "signal_to_noise": q / rv}

"""Optimal portfolio and risk-control tools.

hrp_weights          : hierarchical risk parity (Lopez de Prado 2016) on a covariance matrix
ledoit_wolf_cov      : shrinkage covariance (sklearn)
bm_pass_probability  : Brownian motion with drift m and volatility s per day (in % of the initial balance):
                       P(reach +a before -b) = (1 - exp(-2 m b / s^2)) / (1 - exp(-2 m (a + b) / s^2)) and the expected
                       time to exit, E[T] = (a P - b (1 - P)) / m   (optional stopping on X_t - m t)
policy_mc            : block-bootstrap Monte Carlo of a daily R-return path under a sizing policy
                       (risk per R as a function of the cushion), FP two-phase challenge and funded payouts
"""
import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(X):
    return LedoitWolf().fit(np.asarray(X, float)).covariance_


def hrp_weights(cov):
    cov = np.asarray(cov, float)
    sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    order = leaves_list(linkage(squareform(dist, checks=False), "single"))
    w = np.ones(len(cov))
    clusters = [list(order)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            h = len(c) // 2; a, b = c[:h], c[h:]

            def cvar(ix):
                sub = cov[np.ix_(ix, ix)]; iv = 1 / np.diag(sub); iv /= iv.sum()
                return float(iv @ sub @ iv)
            va, vb = cvar(a), cvar(b)
            al = 1 - va / (va + vb)
            w[a] *= al; w[b] *= 1 - al
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def bm_pass_probability(m, s, a, b):
    if abs(m) < 1e-12:
        p = b / (a + b)
        return p, a * b / s ** 2
    k = 2 * m / s ** 2
    p = (1 - math.exp(-k * b)) / (1 - math.exp(-k * (a + b)))
    return p, (a * p - b * (1 - p)) / m


def policy_mc(R: np.ndarray, risk_fn, n_paths=4000, block=10, horizon=750, targets=(8.0, 5.0), floor=8.0, daily_lim=3.0,
              seed=0):
    """R: historical daily P&L in R units at full size (R = 0.5 % of the initial balance when risk_fn = 1).
    risk_fn(dd_pct) -> size multiplier in [0, 1] given the current drawdown from the phase start (%).
    A phase starts at 0 %; pass when cumulative >= target, fail when <= -floor or a day loses >= daily_lim.
    Returns P(pass both), P(fail), median days to pass (conditional)."""
    rng = np.random.default_rng(seed)
    n = len(R)
    res = {"pass": 0, "fail": 0, "timeout": 0}; days = []
    for _ in range(n_paths):
        t = 0; ok = True; total_days = 0
        for tgt in targets:
            eq = 0.0; phase_days = 0
            while True:
                if t % block == 0:
                    start = int(rng.integers(0, n - block))
                r = R[start + t % block]; t += 1; phase_days += 1
                mult = risk_fn(max(0.0, -eq))
                day = 0.5 * mult * r                     # % of initial balance
                if day <= -daily_lim or eq + day <= -floor:
                    ok = False; res["fail"] += 1; break
                eq += day
                if eq >= tgt:
                    break
                if phase_days >= horizon:
                    ok = False; res["timeout"] += 1; break
            total_days += phase_days
            if not ok:
                break
        if ok:
            res["pass"] += 1; days.append(total_days)
    return {"p_pass": res["pass"] / n_paths, "p_fail": res["fail"] / n_paths, "p_timeout": res["timeout"] / n_paths,
            "median_days": float(np.median(days)) if days else None}

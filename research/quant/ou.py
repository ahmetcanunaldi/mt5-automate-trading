"""Ornstein-Uhlenbeck toolkit: dX = theta (mu - X) dt + sigma dW.

fit_ou            : exact-discretization MLE (AR(1) regression) -> theta, mu, sigma, half-life
ou_expected_time  : E[first passage time a -> m] for a zero-mean OU (a < m), via
                    E[T] = sqrt(pi)/theta * int_{a sqrt(theta)/sigma}^{m sqrt(theta)/sigma} exp(z^2) (1 + erf(z)) dz
bertram_threshold : symmetric entry band a* (enter at -a, exit at +a ... here: enter at |X| >= a, exit at 0 or at the
                    opposite band) maximizing expected profit per unit time (a_exit - a_entry - cost) / E[T]
                    (Bertram 2010, Analytic solutions for optimal statistical arbitrage trading)
simulate_ou       : exact simulation (tests)
"""
import math

import numpy as np
from scipy import integrate, optimize, special


def fit_ou(x, dt=1.0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    X, Y = x[:-1], x[1:]
    b, a = np.polyfit(X, Y, 1)
    b = min(max(b, 1e-6), 1 - 1e-9)
    theta = -math.log(b) / dt
    mu = a / (1 - b)
    resid = Y - (a + b * X)
    s2 = resid.var(ddof=2)
    sigma = math.sqrt(s2 * 2 * theta / (1 - b ** 2))
    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": math.log(2) / theta, "b": b,
            "sigma_eq": sigma / math.sqrt(2 * theta)}


def ou_expected_time(a, m, theta, sigma):
    """Expected time for a zero-mean OU started at a to first reach m (a < m)."""
    lo, hi = a * math.sqrt(theta) / sigma, m * math.sqrt(theta) / sigma
    f = lambda z: math.exp(z * z) * (1 + special.erf(z))  # noqa: E731
    val, _ = integrate.quad(f, lo, hi, limit=200)
    return math.sqrt(math.pi) / theta * val


def bertram_threshold(theta, sigma, cost):
    """Optimal symmetric band for a zero-mean OU: enter short at +a (long at -a), exit at the opposite band -a (+a).
    Profit per round trip 2a - cost; cycle time = E[T_{-a->a}] + E[T_{a->-a}] = 2 E[T_{-a->a}] (symmetry), i.e. one
    full cycle yields two trades. Maximizes (2a - cost) / E[T_{-a->a}]. Returns a*, rate, expected trade time."""
    s_eq = sigma / math.sqrt(2 * theta)
    rate = lambda a: -(2 * a - cost) / ou_expected_time(-a, a, theta, sigma)  # noqa: E731
    res = optimize.minimize_scalar(rate, bounds=(max(cost / 2 * 1.0001, 1e-9), 4 * s_eq), method="bounded")
    a = float(res.x)
    return {"a": a, "a_in_sigma_eq": a / s_eq, "rate": -float(res.fun), "E_T": ou_expected_time(-a, a, theta, sigma)}


def simulate_ou(theta, mu, sigma, n, dt=1.0, x0=None, seed=0):
    rng = np.random.default_rng(seed)
    b = math.exp(-theta * dt); sd = sigma * math.sqrt((1 - b * b) / (2 * theta))
    x = np.empty(n); x[0] = mu if x0 is None else x0
    for t in range(1, n):
        x[t] = mu + (x[t - 1] - mu) * b + sd * rng.standard_normal()
    return x

import math

import numpy as np

from research.quant import coint, ou


def test_fit_ou_recovers_parameters():
    x = ou.simulate_ou(theta=0.05, mu=1.0, sigma=0.2, n=40000, seed=1)
    p = ou.fit_ou(x)
    assert abs(p["theta"] - 0.05) < 0.01 and abs(p["mu"] - 1.0) < 0.1 and abs(p["sigma"] - 0.2) < 0.01
    assert abs(p["half_life"] - math.log(2) / 0.05) < 3


def test_expected_passage_time_matches_simulation():
    theta, sigma, a = 0.1, 0.3, 0.3
    ana = ou.ou_expected_time(-a, 0.0, theta, sigma)
    rng = np.random.default_rng(2); dt = 0.002; b = math.exp(-theta * dt); sd = sigma * math.sqrt((1 - b * b) / (2 * theta))
    n = 3000; x = np.full(n, -a); t = np.zeros(n); alive = np.ones(n, bool)
    while alive.any():                                   # vectorized paths, fine monitoring (small discretization bias)
        x[alive] = x[alive] * b + sd * rng.standard_normal(alive.sum()); t[alive] += dt; alive &= x < 0
    T = t
    assert abs(np.mean(T) - ana) / ana < 0.1


def test_bertram_threshold_rises_with_cost():
    lo = ou.bertram_threshold(0.1, 0.3, cost=0.01)["a_in_sigma_eq"]
    hi = ou.bertram_threshold(0.1, 0.3, cost=0.3)["a_in_sigma_eq"]
    assert 0 < lo < hi


def test_kalman_hedge_tracks_beta():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 0.01, 5000))
    beta = np.r_[np.full(2500, 1.2), np.full(2500, 0.8)]
    y = 0.1 + beta * x + ou.simulate_ou(0.1, 0, 0.002, 5000, seed=4)
    k = coint.kalman_hedge(y, x, delta=1e-4, r_var=1e-5)
    assert abs(k.beta.iloc[2400] - 1.2) < 0.15 and abs(k.beta.iloc[-1] - 0.8) < 0.15
    eg = coint.engle_granger(y[:2500], x[:2500])
    assert eg["adf_p"] < 0.01

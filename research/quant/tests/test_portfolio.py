import numpy as np

from research.quant import portfolio as P


def test_bm_pass_probability_limits_and_mc():
    p0, _ = P.bm_pass_probability(0.0, 1.0, 8, 8)
    assert abs(p0 - 0.5) < 1e-9
    p, et = P.bm_pass_probability(0.1, 1.0, 8, 8)
    rng = np.random.default_rng(0); hits = 0; T = []
    for _ in range(3000):
        x, t = 0.0, 0
        while -8 < x < 8:
            x += 0.1 + rng.standard_normal(); t += 1
        hits += x >= 8; T.append(t)
    assert abs(hits / 3000 - p) < 0.04
    assert abs(np.mean(T) - et) / et < 0.15


def test_smaller_bets_raise_pass_probability_with_positive_drift():
    p_big, _ = P.bm_pass_probability(0.1, 1.0, 8, 8)
    p_small, t_small = P.bm_pass_probability(0.05, 0.5, 8, 8)         # half size: m and s halve
    assert p_small > p_big


def test_hrp_weights_sum_to_one_and_favor_low_vol():
    cov = np.diag([0.01, 0.04, 0.09])
    w = P.hrp_weights(cov)
    assert abs(w.sum() - 1) < 1e-9 and w[0] > w[1] > w[2]

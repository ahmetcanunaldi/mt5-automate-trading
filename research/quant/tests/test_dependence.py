import numpy as np
import pandas as pd

from research.quant import dependence as D


def ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_t(5, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_variance_ratio_and_autocorr():
    iid = np.random.default_rng(0).standard_t(5, 20000)
    vr, z = D.variance_ratio(iid, 4)
    assert abs(vr - 1) < 0.05 and abs(z) < 3
    mom = ar1(0.1, 20000, 1)
    vr, z = D.variance_ratio(mom, 4)
    assert vr > 1.1 and z > 5
    rho, t = D.autocorr(ar1(-0.1, 20000, 2))
    assert rho < -0.07 and t < -5


def test_dfa_hurst_iid_near_half():
    h = D.dfa_hurst(np.random.default_rng(3).normal(size=20000))
    assert 0.44 < h < 0.56


def test_mi_and_te_detect_dependence():
    rng = np.random.default_rng(4)
    x = rng.normal(size=20000)
    y = np.r_[0, 0.5 * x[:-1]] + rng.normal(size=20000)      # x leads y by one step
    assert D.transfer_entropy(x, y) > 5 * D.transfer_entropy(y, x)
    assert D.mutual_info(x[:-1], y[1:]) > 10 * D.mutual_info(rng.normal(size=19999), y[1:])
    ll = D.lead_lag(pd.Series(x), pd.Series(y), 2)
    assert ll[1] > 0.3 and abs(ll[-1]) < 0.05

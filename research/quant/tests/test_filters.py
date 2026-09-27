import numpy as np

from research.quant import filters as F


def test_hmm_filter_recovers_regimes_causally():
    rng = np.random.default_rng(0)
    n = 4000
    A = np.array([[0.98, 0.02], [0.03, 0.97]])
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    mu = np.array([-0.002, 0.001]); sd = np.array([0.02, 0.008])
    x = (mu[s] + sd[s] * rng.standard_normal(n))[:, None]
    out = F.hmm_fit_filter(x[:3000], x, 2)
    p_bull = out["alpha"][3000:, 1]
    acc = ((p_bull > 0.5).astype(int) == s[3000:]).mean()
    assert acc > 0.8
    assert out["A"][0, 0] > 0.9 and out["A"][1, 1] > 0.9


def test_kalman_drift_tracks_slow_mean():
    rng = np.random.default_rng(1)
    n = 3000
    mu = np.cumsum(rng.normal(0, 0.0003, n))
    r = mu + rng.normal(0, 0.01, n)
    out, par = F.kalman_drift(r[:2000], r)
    assert np.corrcoef(out.mu[2000:], mu[2000:])[0, 1] > 0.6
    assert par["signal_to_noise"] < 0.01

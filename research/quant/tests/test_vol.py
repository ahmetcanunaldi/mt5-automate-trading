import numpy as np
import pandas as pd

from research.quant import vol as V


def test_rough_hurst_recovers_random_walk_log_vol():
    rng = np.random.default_rng(0)
    log_sig = np.cumsum(rng.normal(0, 0.05, 4000)) - 5          # log-vol random walk -> H = 0.5
    rv = pd.Series(np.exp(2 * log_sig), index=pd.date_range("2010-01-01", periods=4000, freq="B"))
    h, _ = V.rough_hurst(rv)
    assert 0.42 < h < 0.58


def test_har_beats_naive_on_persistent_vol():
    rng = np.random.default_rng(1)
    n = 3000
    ls = np.zeros(n)
    for t in range(1, n):
        ls[t] = 0.97 * ls[t - 1] + rng.normal(0, 0.15)
    rv = pd.Series(np.exp(ls - 9) * rng.chisquare(20, n) / 20, index=pd.date_range("2012-01-01", periods=n, freq="B"))
    X, y = V.har_design(rv)
    tr = rv.index < "2020-01-01"
    b, s2 = V.har_fit(X[tr], y[tr])
    f = V.har_forecast(X, b, s2).shift(1)[~tr]
    assert V.qlike(rv[~tr], f) < V.qlike(rv[~tr], rv.shift(1)[~tr])
    assert V.mz_r2(rv[~tr], f) > 0.2


def test_garch_recovers_persistence():
    rng = np.random.default_rng(2)
    n = 4000; w, a, b = 0.02, 0.08, 0.9
    r = np.zeros(n); h = np.full(n, w / (1 - a - b))
    for t in range(1, n):
        h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(8) / np.sqrt(8 / 6)
    s = pd.Series(r / 100, index=pd.date_range("2008-01-01", periods=n, freq="B"))
    _, p = V.garch_forecasts(s, s.index[3000], kind="garch")
    assert 0.8 < p["alpha[1]"] + p["beta[1]"] < 1.0

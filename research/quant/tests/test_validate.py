import numpy as np
import pandas as pd

from research.quant import validate as V


def test_psr_and_dsr_behave():
    rng = np.random.default_rng(1)
    good = rng.normal(0.1, 1.0, 2000)            # per-period SR 0.1 -> annualized ~1.6
    noise = rng.normal(0.0, 1.0, 2000)
    assert V.probabilistic_sharpe(good) > 0.99
    assert 0.05 < V.probabilistic_sharpe(noise) < 0.95 or abs(V.probabilistic_sharpe(noise) - 0.5) < 0.45
    # deflating for many trials lowers the probability
    assert V.deflated_sharpe(good, 1000) < V.deflated_sharpe(good, 1)
    assert abs(V.deflated_sharpe(good, 1) - V.probabilistic_sharpe(good)) < 1e-9


def test_pbo_noise_vs_signal():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (1600, 30))
    assert V.pbo_cscv(noise, 10)["pbo"] > 0.4          # noise: IS winner is not an OOS winner (CSCV halves are complementary -> often > 0.5)
    sig = noise.copy(); sig[:, 0] += 0.3            # one clearly superior configuration
    assert V.pbo_cscv(sig, 10)["pbo"] < 0.1


def test_spa_noise_not_significant_and_signal_significant():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, (1500, 20))
    assert V.spa_test(noise, B=300)["p_consistent"] > 0.05
    sig = noise.copy(); sig[:, 3] += 0.15
    assert V.spa_test(sig, B=300)["p_consistent"] < 0.05


def test_placebo_detects_real_link_only():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2018-01-01", periods=1500, freq="B")
    x = pd.Series(rng.normal(size=1500), index=idx)
    y = 0.3 * x.shift(1).fillna(0) + rng.normal(size=1500)   # y_t depends on x_{t-1}
    stat = lambda s: float(np.corrcoef(x.shift(1).fillna(0), s)[0, 1])  # noqa: E731
    assert V.placebo(stat, pd.Series(y, index=idx), n=100)["p"] < 0.05
    stat_noise = lambda s: float(np.corrcoef(x.shift(1).fillna(0), s)[0, 1])  # noqa: E731
    assert V.placebo(stat_noise, pd.Series(rng.normal(size=1500), index=idx), n=100)["p"] > 0.01


def test_yearly_splits_no_overlap():
    idx = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    for y, tr, te in V.yearly_splits(idx, 2020, 2024, embargo="5D"):
        assert idx[tr].max() < idx[te].min() - pd.Timedelta("4D")

import numpy as np

from research.quant import jumps as J


def test_lee_mykland_finds_planted_jumps():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1e-4, 20000)
    idx = rng.choice(np.arange(500, 20000), 20, replace=False)
    r[idx] += np.sign(rng.normal(size=20)) * 1.5e-3                # 15 sigma jumps
    L, crit = J.lee_mykland(r)
    found = set(np.flatnonzero(np.abs(L) > crit))
    assert len(found & set(idx)) >= 18
    assert len(found - set(idx)) <= 5


def test_hawkes_fit_recovers_branching_ratio():
    t = J.hawkes_sim(0.5, 0.6, 1.2, 4000, seed=1)
    p = J.hawkes_fit(t, 4000)
    assert abs(p["branching"] - 0.5) < 0.12
    assert abs(p["mu"] - 0.5) < 0.15

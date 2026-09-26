import numpy as np
import pandas as pd

from research.features import atr
from research.structure import structure_frame, zigzag


def _bars(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 2000 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2025-01-06 01:00", periods=n, freq="5min")
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.random(n); l = np.minimum(o, c) - rng.random(n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_structure_is_causal():
    b = _bars()
    a = atr(b, 14)
    full, _ = structure_frame(b, a, 2.0, "s")
    cut = 2000
    part, _ = structure_frame(b.iloc[:cut], a.iloc[:cut], 2.0, "s")
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_zigzag_alternates_and_confirms_after_pivot():
    b = _bars()
    a = atr(b, 14).to_numpy()
    ci, pi, pr, kd = zigzag(b.high.to_numpy(), b.low.to_numpy(), a, 2.0)
    assert len(kd) > 20
    assert np.all(kd[1:] != kd[:-1])          # H, L, H, L ...
    assert np.all(pi < ci)                    # pivot known only later

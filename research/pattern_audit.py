"""EXP-098: overfitting audit of the pattern leg.
(1) what is selected: top patterns per symbol for 2026 (trained on all earlier years), in words
(2) selection stability: share of year-Y patterns that were also selected in Y-1
(3) placebo: the same walk-forward on next-day returns shuffled within each year (100 permutations) -> distribution
    of the OOS 'vote+ minus rest' spread under the null; the real spread's percentile = p-value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.pattern_robust import FILES, prep  # noqa: E402


def wf(d, yR, M, keep_sel=False):
    years = d.index.year.to_numpy(); valid = np.isfinite(yR); up, rest, sel = [], [], {}
    for Y in range(years.min() + 5, years.max() + 1):
        tr = valid & (years < Y)
        ytr = np.nan_to_num(yR[tr]); ytr = ytr - ytr.mean(); X = M[:, tr]
        n = X.sum(1); mu = np.where(n > 0, X @ ytr / np.maximum(n, 1), 0)
        var = np.where(n > 1, X @ ytr ** 2 / np.maximum(n, 1) - mu ** 2, np.inf)
        t = mu / np.sqrt(np.maximum(var, 1e-12) / np.maximum(n, 1))
        pick = (np.abs(t) >= 3.0) & (n >= 60)
        if keep_sel:
            sel[Y] = {i: (float(mu[i]), float(t[i]), int(n[i])) for i in np.flatnonzero(pick)}
        te = valid & (years == Y); v = (np.sign(mu) * pick) @ M[:, te]
        up += list(yR[te][v > 0]); rest += list(yR[te][v <= 0])
    up, rest = np.array(up), np.array(rest)
    return (up.mean() - rest.mean() if len(up) else 0.0), len(up), sel


if __name__ == "__main__":
    from research import pattern_mine
    rng = np.random.default_rng(11)
    for sym in FILES:
        d, yR, M = prep(sym)
        names = list(pattern_mine.patterns(pattern_mine.literals(d)))
        real, nup, sel = wf(d, yR, M, keep_sel=True)
        ys = sorted(sel)
        stab = [len(set(sel[y]) & set(sel[y - 1])) / max(len(sel[y]), 1) for y in ys[1:]]
        print(f"\n=== {sym}: OOS spread vote+ minus rest {real:+.3f} R (vote+ days {nup})")
        print(f"  patterns selected per year: {[len(sel[y]) for y in ys]}")
        print(f"  share also selected the year before: {[round(s, 2) for s in stab]}")
        top = sorted(sel[ys[-1]].items(), key=lambda kv: -abs(kv[1][1]))[:8]
        print(f"  {ys[-1]} selection (pattern, excess R per day vs average day, t, n):")
        for i, (mu, t, n) in top:
            print(f"     {names[i]:<40} {mu:+.3f}  t {t:+.2f}  n {n}")
        null = []
        years = d.index.year.to_numpy()
        for _ in range(100):
            yp = yR.copy()
            for y in np.unique(years):
                m = (years == y) & np.isfinite(yR)
                yp[m] = rng.permutation(yR[m])
            null.append(wf(d, yp, M)[0])
        null = np.array(null)
        print(f"  placebo (shuffled returns, 100x): spread mean {null.mean():+.3f}, 95th pct {np.percentile(null, 95):+.3f}, "
              f"max {null.max():+.3f} -> p = {(null >= real).mean():.2f}")

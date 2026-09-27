"""EXP-095: robustness of the walk-forward pattern selection (EXP-089/090) and a ridge alternative.
(a) grid T_SEL in {2.5, 3.0, 3.5} x N_MIN in {40, 60, 100}: OOS next-day long R on vote+ days vs no-vote days
(b) ridge regression on all pattern indicators (walk-forward, alpha by train CV-free heuristic = 1e3..1e5),
    prediction top 20 % days -> long; OOS mean R vs the other days."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab, pattern_mine  # noqa: E402
from research.features import atr  # noqa: E402

FILES = {"XAUUSD": "XAUUSD_D1_2007", "NAS100": "NAS100_D1_long", "DJ30": "DJ30_D1_long"}


def prep(sym):
    d = pd.read_parquet(lab.DATA / f"{FILES[sym]}.parquet")[["open", "high", "low", "close"]]
    d = d[(d.index.dayofweek < 5) & (d.high > d.low)]
    a_bps = atr(d, 14) / d.close * 1e4
    yR = ((np.log(d.close / d.open) * 1e4).shift(-1) / a_bps).to_numpy()
    P = pattern_mine.patterns(pattern_mine.literals(d))
    names = list(P)
    return d, yR, np.vstack([P[n] for n in names]).astype(np.float32)


def vote_oos(d, yR, M, tsel, nmin):
    years = d.index.year.to_numpy(); valid = np.isfinite(yR); up, rest = [], []
    for Y in range(years.min() + 5, years.max() + 1):
        tr = valid & (years < Y)
        ytr = np.nan_to_num(yR[tr]); ytr = ytr - ytr.mean(); X = M[:, tr]
        n = X.sum(1); mu = np.where(n > 0, X @ ytr / np.maximum(n, 1), 0)
        var = np.where(n > 1, X @ ytr ** 2 / np.maximum(n, 1) - mu ** 2, np.inf)
        t = mu / np.sqrt(np.maximum(var, 1e-12) / np.maximum(n, 1))
        s = np.sign(mu) * ((np.abs(t) >= tsel) & (n >= nmin))
        te = valid & (years == Y); v = s @ M[:, te]
        up += list(yR[te][v > 0]); rest += list(yR[te][v <= 0])
    up, rest = np.array(up), np.array(rest)
    return up.mean(), len(up), rest.mean(), (up.mean() - rest.mean()) / np.sqrt(up.var() / len(up) + rest.var() / len(rest))


def ridge_oos(d, yR, M, alpha):
    years = d.index.year.to_numpy(); valid = np.isfinite(yR); top, rest = [], []
    for Y in range(years.min() + 5, years.max() + 1):
        tr = valid & (years < Y)
        X = M[:, tr].T; y = np.nan_to_num(yR[tr]); y = y - y.mean()
        mu = X.mean(0); Xc = X - mu
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1], dtype=np.float32), Xc.T @ y)
        te = valid & (years == Y)
        p = (M[:, te].T - mu) @ w
        cut = np.quantile((X - mu) @ w, 0.8)                        # threshold from the train distribution
        top += list(yR[te][p > cut]); rest += list(yR[te][p <= cut])
    top, rest = np.array(top), np.array(rest)
    return top.mean(), len(top), rest.mean(), (top.mean() - rest.mean()) / np.sqrt(top.var() / len(top) + rest.var() / len(rest))


if __name__ == "__main__":
    for sym in FILES:
        d, yR, M = prep(sym)
        print(f"=== {sym} ({M.shape[0]} patterns)")
        for tsel in (2.5, 3.0, 3.5):
            print("  vote", " | ".join(f"T{tsel} N{nmin}: +{a:+.3f} (n {n}) vs {b:+.3f} t {t:+.2f}"
                                       for nmin in (40, 60, 100) for a, n, b, t in [vote_oos(d, yR, M, tsel, nmin)]))
        # ridge on singles+pairs only (triples make the system too large)
        keep = np.arange(min(M.shape[0], 700))
        for alpha in (1e3, 1e4, 1e5):
            a, n, b, t = ridge_oos(d, yR, M[keep], alpha)
            print(f"  ridge alpha {alpha:.0e}: top20% {a:+.3f} (n {n}) vs rest {b:+.3f}, t {t:+.2f}")

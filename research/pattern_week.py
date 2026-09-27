"""EXP-097: pattern mining with a rest-of-week target (next day's open -> Friday close, entries Mon-Thu decisions
... i.e. decision days Fri->Mon excluded), walk-forward, demeaned, same literals/patterns as EXP-089.
Evaluation in ATR units per day held (target / holding days) to compare with the next-day version."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab, pattern_mine  # noqa: E402
from research.features import atr  # noqa: E402

FILES = {"XAUUSD": "XAUUSD_D1_2007", "NAS100": "NAS100_D1_long", "DJ30": "DJ30_D1_long", "GER40": "GER40_D1_long"}

if __name__ == "__main__":
    for sym, f in FILES.items():
        d = pd.read_parquet(lab.DATA / f"{f}.parquet")[["open", "high", "low", "close"]]
        d = d[(d.index.dayofweek < 5) & (d.high > d.low)]
        a_bps = atr(d, 14) / d.close * 1e4
        wk = d.index.to_period("W-FRI")
        fri_close = d.close.groupby(wk).transform("last")
        nxt_open = d.open.shift(-1)
        same_week = pd.Series(wk, index=d.index).shift(-1) == pd.Series(wk, index=d.index)
        ndays = (pd.Series(1, index=d.index).groupby(wk).cumcount(ascending=False)).astype(float)   # days left after t
        y = (np.log(fri_close / nxt_open) * 1e4).where(same_week & (ndays > 0))
        yR = (y / a_bps / ndays).to_numpy()                       # per held day
        P = pattern_mine.patterns(pattern_mine.literals(d))
        names = list(P); M = np.vstack([P[n] for n in names]).astype(np.float32)
        years = d.index.year.to_numpy(); valid = np.isfinite(yR)
        up, rest, dn = [], [], []
        for Y in range(years.min() + 5, years.max() + 1):
            tr = valid & (years < Y)
            ytr = np.nan_to_num(yR[tr]); ytr = ytr - ytr.mean(); X = M[:, tr]
            n = X.sum(1); mu = np.where(n > 0, X @ ytr / np.maximum(n, 1), 0)
            var = np.where(n > 1, X @ ytr ** 2 / np.maximum(n, 1) - mu ** 2, np.inf)
            t = mu / np.sqrt(np.maximum(var, 1e-12) / np.maximum(n, 1))
            s = np.sign(mu) * ((np.abs(t) >= 3.0) & (n >= 60))
            te = valid & (years == Y); v = s @ M[:, te]
            up += list(yR[te][v > 0]); dn += list(yR[te][v < 0]); rest += list(yR[te][v == 0])
        up, dn, rest = map(np.array, (up, dn, rest))
        tt = (up.mean() - rest.mean()) / np.sqrt(up.var() / len(up) + rest.var() / len(rest)) if len(up) > 2 else np.nan
        print(f"{sym}: OOS per-day R: vote+ {up.mean() if len(up) else np.nan:+.3f} (n {len(up)}), vote- "
              f"{dn.mean() if len(dn) else np.nan:+.3f} (n {len(dn)}), none {rest.mean():+.3f} (n {len(rest)}), vote+ vs none t {tt:+.2f}")

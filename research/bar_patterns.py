"""EXP-093: intraday candlestick pattern mining (H1 / H4 bars from M1 2018-2026), yearly walk-forward, demeaned.
Literals per bar t (and t-1): direction, close vs previous close, CLV terciles, range / ATR(14 bars), gap,
20-bar position, plus upper/lower wick share (> 0.5 of range = pin), body share (< 0.2 = doji, > 0.7 = marubozu),
and the hour of bar t+1 (server). Patterns: singles, pairs, triples (pair x t-1 literal / pair x hour).
Target: bar t+1 open->close in ATR units, cost subtracted in the evaluation. Train = all years before Y (>= 3),
keep |t| >= 3.5 (more tests), n >= 150. OOS: vote+ / vote- / none next-bar R and a long-short spread."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab, pattern_mine, symbols  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

COST_BPS = {"XAUUSD": 1.5, "NAS100": 1.3, "DJ30": 1.2}


def literals(b):
    L = pattern_mine.literals(b)
    L = L[[c for c in L.columns if not c.startswith("next")]]
    rng = (b.high - b.low).replace(0, np.nan)
    up_w = (b.high - b[["open", "close"]].max(axis=1)) / rng
    lo_w = (b[["open", "close"]].min(axis=1) - b.low) / rng
    body = (b.close - b.open).abs() / rng
    X = {"pin_up": up_w > 0.5, "pin_lo": lo_w > 0.5, "doji": body < 0.2, "maru": body > 0.7}
    X = {k: v.fillna(False).astype(bool) for k, v in X.items()}
    X.update({f"{k}[-1]": v.shift(1).fillna(False).astype(bool) for k, v in list(X.items())})
    L = pd.concat([L, pd.DataFrame(X)], axis=1)
    nh = pd.Series(np.roll(b.index.hour, -1), index=b.index)
    for h in sorted(np.unique(b.index.hour)):
        L[f"next_h{h:02d}"] = nh == h
    return L


def patterns(L):
    today = [c for c in L.columns if "[-1]" not in c and not c.startswith("next")]
    prev = [c for c in L.columns if "[-1]" in c]
    hours = [c for c in L.columns if c.startswith("next")]
    P = {c: L[c].to_numpy() for c in L.columns}
    fam = lambda c: c.replace("[-1]", "").split("_")[0]  # noqa: E731
    for a, b in itertools.combinations(today + prev + hours, 2):
        if (fam(a) == fam(b) and ("[-1]" in a) == ("[-1]" in b)) or (a.startswith("next") and b.startswith("next")):
            continue
        P[f"{a}&{b}"] = P[a] & P[b]
    pairs = [(a, b) for a, b in itertools.combinations(today, 2) if fam(a) != fam(b)]
    for (a, b), c in itertools.product(pairs, prev + hours):
        P[f"{a}&{b}&{c}"] = P[a] & P[b] & P[c]
    return P


def run(sym, tf):
    m1 = symbols.load_m1(sym).loc["2018":"2026-09-25"]
    b = m1.resample(tf).agg(AGG).dropna(subset=["open"])
    b = b[b.index.dayofweek < 5]
    a_bps = atr(b, 14) / b.close * 1e4
    nxt_same_day = pd.Series(b.index.normalize(), index=b.index).shift(-1) == b.index.normalize()
    yR = ((np.log(b.close / b.open) * 1e4).shift(-1) / a_bps).where(nxt_same_day).to_numpy()
    cR = (COST_BPS[sym] / a_bps).to_numpy()
    P = patterns(literals(b))
    names = list(P); M = np.vstack([P[n] for n in names]).astype(np.float32)
    years = b.index.year.to_numpy(); valid = np.isfinite(yR)
    out, sel = [], None
    for Y in range(2021, 2027):
        tr = valid & (years < Y)
        ytr = np.nan_to_num(yR[tr]).astype(np.float32); ytr = ytr - ytr.mean()
        Xtr = M[:, tr]
        n = Xtr.sum(1); mu = np.where(n > 0, (Xtr @ ytr) / np.maximum(n, 1), 0)
        var = np.where(n > 1, (Xtr @ (ytr ** 2)) / np.maximum(n, 1) - mu ** 2, np.inf)
        t = mu / np.sqrt(np.maximum(var, 1e-12) / np.maximum(n, 1))
        pick = (np.abs(t) >= 3.5) & (n >= 150)
        sgn = (np.sign(mu) * pick).astype(np.float32)
        sel = sorted([(names[i], round(float(mu[i]), 3), round(float(t[i]), 2), int(n[i])) for i in np.flatnonzero(pick)],
                     key=lambda z: -abs(z[2]))
        te = valid & (years == Y)
        vote = sgn @ M[:, te]
        idx = np.flatnonzero(te)
        out.append(pd.DataFrame({"dir": np.sign(vote).astype(int), "R_long": yR[idx], "cost": cR[idx],
                                 "n_sel": int(pick.sum())}, index=b.index[idx]))
    return pd.concat(out), sel, len(names)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    (lab.REPORTS / "EXP-093").mkdir(exist_ok=True)
    for sym in ("XAUUSD", "NAS100", "DJ30"):
        for tf in ("4h", "1h"):
            o, sel, npat = run(sym, tf)
            up, dn, fl = o[o.dir > 0], o[o.dir < 0], o[o.dir == 0]
            ls = pd.concat([up.R_long - up.cost, -dn.R_long - dn.cost])
            yr = ls.groupby(ls.index.year).mean()
            print(f"{sym} {tf}: {npat} patterns | OOS 2021-26 next-bar R: vote+ {up.R_long.mean():+.3f} (n {len(up)}), "
                  f"vote- {dn.R_long.mean():+.3f} (n {len(dn)}), none {fl.R_long.mean():+.3f} (n {len(fl)}) | traded net "
                  f"{ls.mean():+.3f} R/trade, t {ls.mean() / ls.std() * np.sqrt(len(ls)):+.2f}, yrs+ {(yr > 0).sum()}/{len(yr)}, "
                  f"avg sel {o.n_sel.mean():.0f}", flush=True)
            print("    top:", sel[:6])
            o.to_csv(lab.REPORTS / "EXP-093" / f"oos_{sym}_{tf}.csv")

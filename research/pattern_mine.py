"""EXP-089: daily candlestick-sequence pattern mining with an honest yearly walk-forward.

Each day t is described by discrete literals (known at the close of t): direction (close vs open), close vs
previous close, close location (CLV terciles), range / ATR (narrow / normal / wide), gap at the open
(vs 0.1 ATR), 20-day range position (low / mid / high), plus the same literals for day t-1 and the weekday of t+1.
Patterns = single literals, pairs and triples (triples: pair x next weekday / pair x t-1 literal).
Target: next day's open -> close in ATR units (D1, valid execution), minus the round-trip cost.

Walk-forward: for test year Y, patterns are mined on all years < Y (>= MIN_TRAIN years): keep |t| >= T_SEL,
n >= N_MIN; the trade direction is the sign of the train mean. In year Y every selected pattern that is active on
day t votes; the day is traded in the direction of the vote majority (flat on ties). Reported OOS: mean R per
trade, t, per-year, and the same for "always long" and "always trade the drift sign" baselines."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.features import atr  # noqa: E402

COST_BPS = {"XAUUSD": 1.5, "NAS100": 1.3, "DJ30": 1.2, "GER40": 1.3}
T_SEL, N_MIN, MIN_TRAIN = 3.0, 60, 5
DEMEAN = "--demean" in sys.argv


def literals(d):
    a = atr(d, 14)
    rng = d.high - d.low
    clv = ((d.close - d.low) - (d.high - d.close)) / rng.replace(0, np.nan)
    pos = (d.close - d.low.rolling(20).min()) / (d.high.rolling(20).max() - d.low.rolling(20).min())
    gap = (d.open - d.close.shift(1)) / a.shift(1)
    rr = rng / a.shift(1)
    L = {
        "up": d.close > d.open, "dn": d.close <= d.open,
        "cc_up": d.close > d.close.shift(1), "cc_dn": d.close <= d.close.shift(1),
        "clv_lo": clv < -0.33, "clv_mid": clv.abs() <= 0.33, "clv_hi": clv > 0.33,
        "rng_n": rr < 0.8, "rng_m": (rr >= 0.8) & (rr <= 1.3), "rng_w": rr > 1.3,
        "gap_up": gap > 0.1, "gap_0": gap.abs() <= 0.1, "gap_dn": gap < -0.1,
        "pos_lo": pos < 0.2, "pos_mid": (pos >= 0.2) & (pos <= 0.8), "pos_hi": pos > 0.8,
    }
    L = {k: v.fillna(False).astype(bool) for k, v in L.items()}
    L.update({f"{k}[-1]": v.shift(1).fillna(False).astype(bool) for k, v in list(L.items())})
    nd = pd.Series(np.roll(d.index.dayofweek, -1), index=d.index)
    L.update({f"next_dow{k}": nd == k for k in range(5)})
    return pd.DataFrame(L)


def patterns(L):
    names = list(L.columns)
    today = [n for n in names if "[-1]" not in n and not n.startswith("next")]
    prev = [n for n in names if "[-1]" in n]
    dow = [n for n in names if n.startswith("next")]
    P = {n: L[n].to_numpy() for n in names}
    for a, b in itertools.combinations(today + prev + dow, 2):
        if a.split("_")[0] == b.split("_")[0] and ("[-1]" in a) == ("[-1]" in b):
            continue                                                  # same feature, same day
        if a.startswith("next") and b.startswith("next"):
            continue
        P[f"{a}&{b}"] = P[a] & P[b]
    pairs_today = [(a, b) for a, b in itertools.combinations(today, 2) if a.split("_")[0] != b.split("_")[0]]
    for (a, b), c in itertools.product(pairs_today, prev + dow):
        P[f"{a}&{b}&{c}"] = L[a].to_numpy() & L[b].to_numpy() & L[c].to_numpy()
    return P


def run(sym, d):
    d = d[(d.index.dayofweek < 5) & (d.high > d.low)]
    a_bps = (atr(d, 14) / d.close * 1e4)
    y = ((np.log(d.close / d.open) * 1e4).shift(-1) - 0)          # next day open->close, bps
    yR = (y / a_bps).to_numpy()                                    # in ATR units (ATR known at t)
    cR = (COST_BPS[sym] / a_bps).to_numpy()
    L = literals(d)
    P = patterns(L)
    names = list(P); M = np.vstack([P[n] for n in names])          # patterns x days
    years = d.index.year.to_numpy()
    valid = np.isfinite(yR)
    out, sel_log = [], {}
    for Y in range(years.min() + MIN_TRAIN, years.max() + 1):
        tr = valid & (years < Y)
        Xtr = M[:, tr]; ytr = np.nan_to_num(yR[tr])
        if DEMEAN:
            ytr = ytr - ytr.mean()                                   # conditional vs unconditional drift
        n = Xtr.sum(1); s = Xtr @ ytr; ss = Xtr @ (ytr ** 2)
        mu = np.where(n > 0, s / np.maximum(n, 1), 0); var = np.where(n > 1, ss / np.maximum(n, 1) - mu ** 2, np.inf)
        t = mu / np.sqrt(np.maximum(var, 1e-12) / np.maximum(n, 1))
        pick = (np.abs(t) >= T_SEL) & (n >= N_MIN)
        sgn = np.sign(mu) * pick
        sel_log[Y] = [(names[i], round(float(mu[i]), 3), round(float(t[i]), 2), int(n[i])) for i in np.flatnonzero(pick)]
        te = valid & (years == Y)
        vote = sgn @ M[:, te]
        for k, v in zip(np.flatnonzero(te), vote):
            out.append({"date": d.index[k], "dir": int(np.sign(v)), "R": np.sign(v) * yR[k] - cR[k] * (v != 0),
                        "R_long": yR[k] - cR[k], "n_sel": int(pick.sum())})
    return pd.DataFrame(out).set_index("date") if out else pd.DataFrame(), sel_log, len(names)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    rows = []
    for sym, f in (("XAUUSD", "XAUUSD_D1_2007"), ("NAS100", "NAS100_D1_long"), ("DJ30", "DJ30_D1_long"), ("GER40", "GER40_D1_long")):
        d = pd.read_parquet(lab.DATA / f"{f}.parquet")[["open", "high", "low", "close"]]
        o, sel, npat = run(sym, d)
        if o.empty:
            print(sym, "no OOS trades"); continue
        tt = lambda v: round(v.mean() / v.std() * np.sqrt(len(v)), 2) if len(v) > 2 else 0  # noqa: E731
        up, dn, fl = o[o.dir > 0].R_long, o[o.dir < 0].R_long, o[o.dir == 0].R_long
        diff = up.mean() - dn.mean() if len(dn) else np.nan
        print(f"  {sym} OOS next-day long R: vote+ {up.mean():+.3f} (n {len(up)}), vote- {dn.mean() if len(dn) else float('nan'):+.3f} "
              f"(n {len(dn)}), no vote {fl.mean() if len(fl) else float('nan'):+.3f} (n {len(fl)}); spread +/- {diff:+.3f}, "
              f"Welch t {(up.mean() - dn.mean()) / np.sqrt(up.var() / len(up) + dn.var() / max(len(dn), 1)) if len(dn) > 2 else float('nan'):+.2f}")
        o = o[o.dir != 0]
        yr = o.groupby(o.index.year).R.sum()
        longs, shorts = o[o.dir > 0], o[o.dir < 0]
        rows.append({"sym": sym, "patterns": npat, "oos_years": f"{o.index.year.min()}-{o.index.year.max()}",
                     "trades": len(o), "avgR": round(o.R.mean(), 3), "t": tt(o.R), "R_yr": round(yr.mean(), 1),
                     "yrs_pos": f"{(yr > 0).sum()}/{len(yr)}", "long_n": len(longs), "long_avgR": round(longs.R.mean(), 3),
                     "short_n": len(shorts), "short_avgR": round(shorts.R.mean(), 3) if len(shorts) else None,
                     "same_days_long_avgR": round(o.R_long.mean(), 3), "avg_selected": round(o.n_sel.mean(), 1)})
        print(rows[-1], flush=True)
        last = max(sel); print(f"  {sym} patterns selected for {last} ({len(sel[last])}):",
                               sorted(sel[last], key=lambda z: -abs(z[2]))[:12])
        (lab.REPORTS / "EXP-089").mkdir(exist_ok=True)
        o.to_csv(lab.REPORTS / "EXP-089" / f"oos_{sym}.csv")
    print(pd.DataFrame(rows).to_string(index=False))

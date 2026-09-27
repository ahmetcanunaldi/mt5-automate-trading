"""EXP-092: cross-asset daily pattern mining (walk-forward, demeaned). Same machinery as EXP-089, but the
literal set of the target symbol is extended with the other assets' day-t literals: up/down (close vs prev
close), big move (|ret| > 1 ATR), CLV low/high, 20-day position low/high. Assets: XAUUSD, NAS100, DJ30, GER40,
EURUSD, USDJPY (broker D1). Pairs = own literal x other literal and other x other (same day); no triples.
Targets XAUUSD, NAS100, DJ30 next-day open->close in ATR units. OOS 2018-2026 (train >= 5 years)."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab, pattern_mine  # noqa: E402
from research.features import atr  # noqa: E402

FILES = {"XAUUSD": "XAUUSD_D1_2007", "NAS100": "NAS100_D1_long", "DJ30": "DJ30_D1_long", "GER40": "GER40_D1_long",
         "EURUSD": "EURUSD_D1_long", "USDJPY": "USDJPY_D1_long"}


def load(sym):
    d = pd.read_parquet(lab.DATA / f"{FILES[sym]}.parquet")[["open", "high", "low", "close"]]
    return d[(d.index.dayofweek < 5) & (d.high > d.low)]


def other_literals(sym, d):
    a = atr(d, 14)
    ret = d.close - d.close.shift(1)
    clv = ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)
    pos = (d.close - d.low.rolling(20).min()) / (d.high.rolling(20).max() - d.low.rolling(20).min())
    L = {"up": ret > 0, "dn": ret <= 0, "bigup": ret > a.shift(1), "bigdn": ret < -a.shift(1),
         "clvlo": clv < -0.5, "clvhi": clv > 0.5, "poslo": pos < 0.2, "poshi": pos > 0.8}
    return pd.DataFrame({f"{sym}.{k}": v.fillna(False).astype(bool) for k, v in L.items()})


def build_patterns(target):
    d = load(target)
    own = pattern_mine.literals(d)
    own = own[[c for c in own.columns if "[-1]" not in c and not c.startswith("next")]]
    oth = pd.concat([other_literals(s, load(s)) for s in FILES if s != target], axis=1)
    oth = oth.reindex(d.index).fillna(False).astype(bool)
    P = {c: own[c].to_numpy() for c in own}
    P.update({c: oth[c].to_numpy() for c in oth})
    for a, b in itertools.product(own.columns, oth.columns):
        P[f"{a}&{b}"] = own[a].to_numpy() & oth[b].to_numpy()
    for a, b in itertools.combinations(oth.columns, 2):
        if a.split(".")[0] != b.split(".")[0]:
            P[f"{a}&{b}"] = oth[a].to_numpy() & oth[b].to_numpy()
    return d, P


def run(target):
    d, P = build_patterns(target)
    a_bps = atr(d, 14) / d.close * 1e4
    yR = ((np.log(d.close / d.open) * 1e4).shift(-1) / a_bps).to_numpy()
    cR = (pattern_mine.COST_BPS[target] / a_bps).to_numpy()
    names = list(P); M = np.vstack([P[n] for n in names])
    years = d.index.year.to_numpy(); valid = np.isfinite(yR)
    start = max(2018, years.min() + 5)
    out, last_sel = [], None
    for Y in range(start, years.max() + 1):
        tr = valid & (years < Y) & (years >= 2013)                    # common history for all assets
        Xtr = M[:, tr]; ytr = np.nan_to_num(yR[tr]); ytr = ytr - ytr.mean()
        n = Xtr.sum(1); mu = np.where(n > 0, (Xtr @ ytr) / np.maximum(n, 1), 0)
        var = np.where(n > 1, (Xtr @ ytr ** 2) / np.maximum(n, 1) - mu ** 2, np.inf)
        t = mu / np.sqrt(np.maximum(var, 1e-12) / np.maximum(n, 1))
        pick = (np.abs(t) >= 3.0) & (n >= 60)
        sgn = np.sign(mu) * pick
        last_sel = sorted([(names[i], round(float(mu[i]), 3), round(float(t[i]), 2), int(n[i])) for i in np.flatnonzero(pick)],
                          key=lambda z: -abs(z[2]))
        te = valid & (years == Y)
        vote = sgn @ M[:, te]
        for k, v in zip(np.flatnonzero(te), vote):
            out.append({"date": d.index[k], "dir": int(np.sign(v)), "R_long": yR[k] - cR[k], "n_sel": int(pick.sum())})
    return pd.DataFrame(out).set_index("date"), last_sel, len(names)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    (lab.REPORTS / "EXP-092").mkdir(exist_ok=True)
    for tg in ("XAUUSD", "NAS100", "DJ30"):
        o, sel, npat = run(tg)
        up, dn, fl = o[o.dir > 0].R_long, o[o.dir < 0].R_long, o[o.dir == 0].R_long
        wt = (up.mean() - fl.mean()) / np.sqrt(up.var() / len(up) + fl.var() / len(fl)) if len(up) > 2 else np.nan
        yr = o[o.dir > 0].groupby(o[o.dir > 0].index.year).R_long.mean()
        print(f"{tg}: {npat} patterns, OOS {o.index.year.min()}-{o.index.year.max()}: vote+ {up.mean():+.3f} (n {len(up)}), "
              f"vote- {dn.mean() if len(dn) else float('nan'):+.3f} (n {len(dn)}), none {fl.mean():+.3f} (n {len(fl)}); "
              f"vote+ vs none t {wt:+.2f}; vote+ yearly {yr.round(3).to_dict()}")
        print("   last-year selection:", sel[:10])
        o.to_csv(lab.REPORTS / "EXP-092" / f"oos_{tg}.csv")

"""QM-010: realized higher moments as predictors (DEV 2018-2024, walk-forward, placebo).
Daily from 5-min intraday returns (Amaya, Christoffersen, Jacobs & Vasquez 2015; Barndorff-Nielsen et al. 2010;
Patton & Sheppard 2015):
  RSkew = sqrt(N) sum r^3 / RV^1.5, RKurt = N sum r^4 / RV^2, RS+ / RS- = sum r^2 1{r>0 / r<0},
  SJV = (RS+ - RS-) / RV (signed jump variation share), plus the jump share (RV-BV)/RV.
Targets: next day's open->close return and the next 5 days' close-to-close return, both / HAR sigma.
Per predictor: pooled-by-symbol walk-forward regression (train on years < Y), OOS correlation of prediction and
target, long/short OOS return per trade in bps vs cost, placebo p (predictor shuffled within years)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q, validate as VAL  # noqa: E402
from research.quant.qm001_map import COST  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

OUT = lab.REPORTS / "quant" / "QM-010"
PRED = ["rskew", "rkurt", "sjv", "jump", "rs_neg_share"]


def moments(sym):
    b = Q.bars(sym, "5min")
    r = np.log(b.close).diff()
    same = b.index.normalize() == pd.Series(b.index, index=b.index).shift(1).dt.normalize()
    r = r[same]; day = r.index.normalize()
    g = r.groupby(day)
    N = g.size(); rv = (r ** 2).groupby(day).sum()
    d = pd.DataFrame({"rv": rv, "N": N,
                      "rskew": np.sqrt(N) * (r ** 3).groupby(day).sum() / rv ** 1.5,
                      "rkurt": N * (r ** 4).groupby(day).sum() / rv ** 2,
                      "rsp": (r.clip(lower=0) ** 2).groupby(day).sum(), "rsn": (r.clip(upper=0) ** 2).groupby(day).sum()})
    R = Q.realized(sym)
    d = d.join(R[["bv", "open", "close"]], how="inner")
    d["sjv"] = (d.rsp - d.rsn) / d.rv
    d["rs_neg_share"] = d.rsn / d.rv
    d["jump"] = ((d.rv - d.bv).clip(lower=0) / d.rv)
    sig = np.sqrt(d.rv.ewm(alpha=0.1).mean())                          # simple causal vol scale
    d["y1"] = np.log(d.close / d.open).shift(-1) / sig
    d["y5"] = np.log(d.close.shift(-5) / d.close) / (sig * np.sqrt(5))
    d["ret1_bps"] = np.log(d.close / d.open).shift(-1) * 1e4
    d = d[d.N >= 60]
    for c in PRED:                                                     # robust scaling per symbol (causal-ish: expanding)
        m = d[c].expanding(60).median().shift(1); s = (d[c] - m).abs().expanding(60).median().shift(1) * 1.4826
        d[c] = ((d[c] - m) / s).clip(-5, 5)
    return d.dropna(subset=PRED + ["y1"])


def wf(d, col, ycol):
    pred = pd.Series(np.nan, index=d.index)
    for Y in range(2020, 2025):
        tr = d.index.year < Y; te = d.index.year == Y
        x, y = d.loc[tr, col], d.loc[tr, ycol]
        m = x.notna() & y.notna()
        b = np.polyfit(x[m], y[m], 1)
        pred[te] = np.polyval(b, d.loc[te, col])
    return pred


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    D = {s: moments(s) for s in Q.SYMS}
    rows = []
    for col in PRED:
        for ycol in ("y1", "y5"):
            per = []
            for s, d in D.items():
                p = wf(d, col, ycol)
                ok = p.notna() & d[ycol].notna()
                c = np.corrcoef(p[ok], d.loc[ok, ycol])[0, 1]
                pos = np.sign(p[ok] - p[ok].median())                   # relative to the typical prediction (removes drift)
                pnl = (pos * d.loc[ok, "ret1_bps"] - COST[s]) if ycol == "y1" else None
                per.append({"sym": s, "corr": c, "n": int(ok.sum()),
                            "ls_bps": float(pnl.mean()) if pnl is not None else np.nan})
            P = pd.DataFrame(per)
            # pooled significance: Fisher z of mean correlation
            zbar = np.arctanh(P["corr"]).mean(); se = 1 / np.sqrt(P.n.sum())
            # placebo on the pooled mean correlation: shuffle the predictor within years in every symbol
            rng = np.random.default_rng(0); null = []
            for _ in range(40):
                cs = []
                for s, d in D.items():
                    dd = d.copy()
                    dd[col] = dd.groupby(dd.index.year)[col].transform(lambda v: rng.permutation(v.to_numpy()))
                    p = wf(dd, col, ycol); ok = p.notna() & dd[ycol].notna()
                    cs.append(np.corrcoef(p[ok], dd.loc[ok, ycol])[0, 1])
                null.append(np.mean(cs))
            real = P["corr"].mean()
            row = {"predictor": col, "target": ycol, "mean_oos_corr": round(real, 4), "z_pooled": round(zbar / se, 2),
                   "syms_pos": int((P["corr"] > 0).sum()), "placebo_p": round(float(np.mean(np.array(null) >= real)), 3),
                   "ls_bps_mean(y1)": round(P.ls_bps.mean(), 2) if ycol == "y1" else None,
                   "per_sym_corr": {r.sym: round(r.corr, 3) for r in P.itertuples()}}
            rows.append(row); print(row, flush=True)
            led.log("QM-010", "realized_moments", {"pred": col, "target": ycol}, {"corr": real})
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(T.to_string(index=False))

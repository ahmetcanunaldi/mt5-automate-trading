"""EXP-015: pure direction model — which symmetric barrier (+w*sigma / -w*sigma) is touched first?
Timeouts are dropped from training (they carry no direction information).
Trading: long if p_up > 0.5 + delta, short if p_up < 0.5 - delta, SL = TP = w*sigma (realized R, incl. timeouts
and costs, comes from the matching direction row). Walk-forward identical to EXP-014.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research import lab, metrics  # noqa: E402
from research.engine import Guards  # noqa: E402
from research.ml_walkforward import blocks, feature_cols, model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?", default="EXP-015")
    ap.add_argument("--widths", default="1.5,2.0,3.0")
    ap.add_argument("--deltas", default="0.02,0.04,0.06,0.08")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args()
    global blocks
    if a.long:
        lab.PERIODS["WF"] = ("2020-01-01", "2026-09-26", "M1L")
        from research.ml_walkforward import blocks as _b
        blocks = lambda: _b("2020-01-01", "2026-09-26", months=6)  # noqa: E731
    D = pd.read_parquet(lab.DATA / ("ml_dataset_m5_long.parquet" if a.long else "ml_dataset_m5.parquet"))
    cols = [c for c in feature_cols(D) if c != "dir"]
    L = D[D.dir == 1].set_index("close_time")
    S = D[D.dir == -1].set_index("close_time")
    g = Guards(max_positions=a.K, max_open_risk_pct=0.5 * a.K, max_trades_day=10)
    summary, results, objs = [], {}, {}
    for w in map(float, a.widths.split(",")):
        name = f"VOL_{w}_{w}"
        o = L[f"o_{name}"]
        y = (o == 1).astype(float)
        has = o != 0
        y_all = y
        p = pd.Series(np.nan, index=L.index)
        aucs = []
        for s0, s1 in blocks():
            tr = has & (L.index < s0 - pd.Timedelta("1D"))
            te = (L.index >= s0) & (L.index < s1)
            m = model().fit(L.loc[tr, cols], y[tr])
            p[te] = m.predict_proba(L.loc[te, cols])[:, 1]
            ev = te & has
            aucs.append(round(roc_auc_score(y[ev], p[ev]), 3))
        mask = p.notna()
        yrs = p[mask].index.year
        print("   AUC by year:", {int(y): round(roc_auc_score(y_[h_], p_[h_]), 3) for y in sorted(set(yrs))
                                   for y_, p_, h_ in [(y_all[mask][yrs == y], p[mask][yrs == y], has[mask][yrs == y])]
                                   if h_.sum() > 100})
        # realized R of trading each side, by p decile
        dec = pd.qcut(p[mask], 10, labels=False)
        rl = L.loc[mask, f"R_{name}"].groupby(dec).mean().round(3).to_dict()
        rs = S.loc[mask, f"R_{name}"].reindex(p[mask].index).groupby(dec).mean().round(3).to_dict()
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print(f"\n## w={w}: AUC by block {aucs} mean {np.mean(aucs):.3f}")
        print("   long R by p-decile :", rl)
        print("   short R by p-decile:", rs)
        print("   top features:", list(imp.head(15).index), flush=True)
        sl = L[f"sl_{name}"]
        for dl in map(float, a.deltas.split(",")):
            lo = mask & (p > 0.5 + dl); sh = mask & (p < 0.5 - dl)
            parts = []
            for msk, d in ((lo, 1), (sh, -1)):
                parts.append(pd.DataFrame({"dir": d, "sl": sl[msk].to_numpy(), "tp": sl[msk].to_numpy(),
                                           "hold_min": 240, "be": 0.0, "trail": 0.0, "leg": name},
                                          index=pd.DatetimeIndex(sl[msk].index)))
            sig = pd.concat(parts).sort_index()
            if len(sig) < 20:
                continue
            mm, res = lab.evaluate(sig, "WF", guards=g, label=f"{name}|d{dl}", with_challenge=False)
            ch = metrics.challenge_sim(res, max_days=120)
            key = f"{name}|d{dl}"
            results[key] = mm; objs[key] = res
            summary.append({"w": w, "delta": dl, "signals": len(sig), "n": mm["trades"], "SR": mm["sharpe"],
                            "PF": mm["profit_factor"], "avgR": mm["avg_R"], "net%": mm["net_pct"],
                            "DD%": mm["max_total_dd_pct"], "dDD%": mm["max_daily_dd_pct"],
                            "pass120": ch["pass_rate"], "med_days": ch["median_days_to_pass"]})
            print("   ", lab.fmt(mm), flush=True)
    T = pd.DataFrame(summary)
    print("\n", T.to_string())
    best = T.sort_values("SR", ascending=False).head(3)
    keep = {f"VOL_{r.w}_{r.w}|d{r.delta}": objs[f"VOL_{r.w}_{r.w}|d{r.delta}"] for r in best.itertuples()}
    lab.save_experiment(a.exp, {"widths": a.widths, "deltas": a.deltas, "K": a.K},
                        {k: results[k] for k in keep}, keep)
    T.to_csv(lab.REPORTS / a.exp / "summary.csv", index=False)


if __name__ == "__main__":
    main()

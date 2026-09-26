"""EXP-014: walk-forward LightGBM on the dense M5 dataset with dynamic barriers.

Folds: expanding train (from 2024-12) -> 2-month test blocks 2025-07 .. 2026-09, 1-day purge.
Decision: EV = p * (tp/sl) - (1 - p) - cost_R ; trade the better direction per bar if EV > theta.
Trades are then simulated by the engine on M1 (guards, news, EOD) and reported with equity charts.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research import lab, metrics  # noqa: E402
from research.engine import Guards  # noqa: E402

META = {"bar_time", "close_time", "atr", "close", "me_last_l", "me_last_h"}
WF_START, WF_END = "2025-07-01", "2026-09-26"
lab.PERIODS["WF"] = (WF_START, WF_END, "M1")


def feature_cols(D):
    return [c for c in D.columns if c not in META and not c.startswith(("y_", "R_", "sl_", "tp_", "o_"))]


def blocks(start=WF_START, end=WF_END, months=2):
    edges = pd.date_range(start, end, freq=f"{months}MS")
    edges = list(edges) + [pd.Timestamp(end)]
    return list(zip(edges[:-1], edges[1:]))


def model(seed=0):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0,
                              random_state=seed, verbose=-1)


def walk_forward(D, label, cols):
    y = D[f"y_{label}"]
    ok = D[f"R_{label}"].notna()
    pred = pd.Series(np.nan, index=D.index)
    aucs = []
    for a, b in blocks():
        tr = ok & (D.close_time < a - pd.Timedelta("1D"))
        te = ok & (D.close_time >= a) & (D.close_time < b)
        if te.sum() == 0:
            continue
        m = model().fit(D.loc[tr, cols], y[tr])
        pred[te] = m.predict_proba(D.loc[te, cols])[:, 1]
        aucs.append((str(a.date()), round(roc_auc_score(y[te], pred[te]), 3)))
    return pred, aucs, m


def to_signals(D, label, pred, theta, hold=240):
    sl = D[f"sl_{label}"]; tp = D[f"tp_{label}"]
    cost_r = (0.2 + 0.1 + 0.07) / sl
    ev = pred * (tp / sl) - (1 - pred) - cost_r
    X = pd.DataFrame({"t": D.close_time, "dir": D.dir, "sl": sl, "tp": tp, "ev": ev}).dropna()
    X = X[X.ev > theta]
    X = X.sort_values(["t", "ev"], ascending=[True, False]).drop_duplicates("t")
    s = pd.DataFrame({"dir": X.dir.to_numpy(), "sl": X.sl.to_numpy(), "tp": X.tp.to_numpy(), "hold_min": hold,
                      "be": 0.0, "trail": 0.0, "leg": label, "ev": X.ev.to_numpy()},
                     index=pd.DatetimeIndex(X.t.to_numpy()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?", default="EXP-014")
    ap.add_argument("--labels", default="VOL_1.5_2.25,VOL_2.0_3.0,VOL_3.0_4.5,VOL_2.0_2.0,VOL_3.0_6.0,STRUCT")
    ap.add_argument("--thetas", default="0.0,0.1,0.2,0.3")
    ap.add_argument("--K", type=int, default=1)
    a = ap.parse_args()
    D = pd.read_parquet(lab.DATA / "ml_dataset_m5.parquet")
    cols = feature_cols(D)
    print("features:", len(cols), "rows:", len(D))
    g = Guards(max_positions=a.K, max_open_risk_pct=0.5 * a.K, max_trades_day=10)
    results, objs, summary = {}, {}, []
    for label in a.labels.split(","):
        pred, aucs, last_model = walk_forward(D, label, cols)
        m = pred.notna()
        dec = pd.qcut(pred[m], 10, labels=False, duplicates="drop")
        r_by_dec = D.loc[m, f"R_{label}"].groupby(dec).mean().round(3).to_dict()
        print(f"\n## {label}  AUC by block {aucs}")
        print("   mean realized R by predicted-p decile:", r_by_dec)
        imp = pd.Series(last_model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("   top features (gain):", list(imp.head(15).index))
        for th in map(float, a.thetas.split(",")):
            s = to_signals(D, label, pred, th)
            if len(s) < 20:
                continue
            mm, res = lab.evaluate(s, "WF", guards=g, label=f"{label}|th{th}", with_challenge=False)
            mm.update({k: v for k, v in metrics.challenge_sim(res, max_days=120).items()})
            key = f"{label}|th{th}"
            results[key] = mm; objs[key] = res
            summary.append({"label": label, "theta": th, "n": mm["trades"], "SR": mm["sharpe"], "PF": mm["profit_factor"],
                            "avgR": mm["avg_R"], "net%": mm["net_pct"], "DD%": mm["max_total_dd_pct"],
                            "dDD%": mm["max_daily_dd_pct"], "pass120": mm["pass_rate"],
                            "med_days": mm["median_days_to_pass"], "auc_mean": np.mean([x[1] for x in aucs])})
            print("   ", lab.fmt(mm), flush=True)
    S = pd.DataFrame(summary)
    print("\n", S.to_string())
    best = S.sort_values("SR", ascending=False).head(4)
    keep = {f"{r.label}|th{r.theta}": objs[f"{r.label}|th{r.theta}"] for r in best.itertuples()}
    lab.save_experiment(a.exp, {"labels": a.labels, "thetas": a.thetas, "K": a.K, "features": cols,
                                "folds": [(str(x.date()), str(y.date())) for x, y in blocks()]},
                        {k: v for k, v in results.items() if k in keep}, keep)
    S.to_csv(lab.REPORTS / a.exp / "summary.csv", index=False)


if __name__ == "__main__":
    main()

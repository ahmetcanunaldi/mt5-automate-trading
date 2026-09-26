"""EXP-025: gross vs net edge of the M15 direction model, and the break-even round-trip cost.

Model: LightGBM on the M15 8-year dataset, pure-direction label (symmetric ±w·ATR_M15 first touch,
timeouts dropped), 6-month expanding walk-forward blocks 2020..2026.
For each predicted-probability decile we trade the favoured side and report:
  net R    (spread + 2x0.05 slippage + $7/lot commission, as everywhere)
  gross R  (zero cost)
  break-even cost ($/oz round trip) = mean(gross R * stop$) — the cost level at which the book is flat.
Current modelled cost ≈ spread 0.18–0.23 + slippage 0.10 + commission 0.07 ≈ $0.37/oz.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research import lab  # noqa: E402
from research.ml_walkforward import blocks, feature_cols, model  # noqa: E402

CUR_COST = 0.37


def main():
    D = pd.read_parquet(lab.DATA / "ml_dataset_m15_long.parquet")
    cols = [c for c in feature_cols(D) if c != "dir" and not c.startswith("G_")]
    L = D[D.dir == 1].set_index("close_time")
    S = D[D.dir == -1].set_index("close_time")
    rows = []
    for w in (1.5, 2.0, 3.0):
        name = f"VOL_{w}_{w}"
        o = L[f"o_{name}"]; y = (o == 1).astype(float); has = o != 0
        p = pd.Series(np.nan, index=L.index)
        for s0, s1 in blocks("2020-01-01", "2026-09-26", months=6):
            tr = has & (L.index < s0 - pd.Timedelta("1D"))
            te = (L.index >= s0) & (L.index < s1)
            p[te] = model().fit(L.loc[tr, cols], y[tr]).predict_proba(L.loc[te, cols])[:, 1]
        m = p.notna()
        auc = roc_auc_score(y[m & has], p[m & has])
        # favoured side per row: long if p>0.5 else short; conviction = |p-0.5|
        conv = (p[m] - 0.5).abs()
        side_long = p[m] > 0.5
        net = np.where(side_long, L.loc[m, f"R_{name}"], S.loc[m, f"R_{name}"].reindex(L.index[m]))
        gro = np.where(side_long, L.loc[m, f"G_{name}"], S.loc[m, f"G_{name}"].reindex(L.index[m]))
        sl = L.loc[m, f"sl_{name}"].to_numpy()
        T = pd.DataFrame({"conv": conv.to_numpy(), "net": net, "gross": gro, "sl": sl, "year": L.index[m].year})
        T["q"] = pd.qcut(T.conv, 10, labels=False, duplicates="drop")
        print(f"\n## w={w}  AUC={auc:.3f}  rows={len(T)}")
        g = T.groupby("q").agg(net=("net", "mean"), gross=("gross", "mean"), be_cost=("gross", lambda x: 0),
                               n=("net", "size"))
        g["be_cost"] = T.groupby("q").apply(lambda x: (x.gross * x.sl).mean()).round(3)
        g["stop$"] = T.groupby("q")["sl"].median().round(2)
        print(g.round(3).to_string())
        top = T[T.q >= 8]
        yr = top.groupby("year").apply(lambda x: pd.Series({"gross": x.gross.mean(), "net": x.net.mean(),
                                                               "be_cost": (x.gross * x.sl).mean()})).round(3)
        print("  top-20 % conviction by year:\n", yr.T.to_string())
        rows.append({"w": w, "auc": round(auc, 3), "top20_gross_R": round(top.gross.mean(), 3),
                     "top20_net_R": round(top.net.mean(), 3), "top20_be_cost$": round((top.gross * top.sl).mean(), 3),
                     "all_gross_R": round(T.gross.mean(), 3), "cur_cost$": CUR_COST})
    R = pd.DataFrame(rows)
    print("\n", R.to_string(index=False))
    (lab.REPORTS / "EXP-025").mkdir(exist_ok=True)
    R.to_csv(lab.REPORTS / "EXP-025" / "summary.csv", index=False)


if __name__ == "__main__":
    main()

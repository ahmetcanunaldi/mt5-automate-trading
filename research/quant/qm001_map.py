"""QM-001: predictability map (no trading). DEV data 2018-2024 only.
Per symbol x bar size: variance ratios VR(2/5/10) with robust z*, lag-1 autocorrelation (robust t), DFA Hurst,
binned mutual information (lag return -> next return, excess over a shuffled baseline), and the economic check:
|rho1| x E|r| (the per-trade edge of the best sign-following / fading rule) vs the round-trip cost in bps.
Sub-daily returns are intraday only (no overnight gaps). Then the same by session and by volatility regime, and
cross-asset lead-lag / transfer entropy on M1 and M5."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q, dependence as D  # noqa: E402

COST = {"XAUUSD": 1.5, "XAGUSD": 14.6, "NAS100": 1.3, "DJ30": 1.2, "SP500": 1.2, "GER40": 1.3, "EURUSD": 2.2, "USDJPY": 2.2}
TFS = ["1min", "5min", "15min", "1h", "4h", "1D"]
OUT = lab.REPORTS / "quant" / "QM-001"


def row(sym, tf, r, tag="all"):
    r = r.dropna()
    if len(r) < 500:
        return None
    x = r.to_numpy() * 1e4
    rho, t = D.autocorr(x, 1)
    vr2, z2 = D.variance_ratio(x, 2); vr5, z5 = D.variance_ratio(x, 5); vr10, z10 = D.variance_ratio(x, 10)
    mi = D.mutual_info(x[:-1], x[1:]); mi0 = D.mutual_info(np.random.default_rng(0).permutation(x[:-1]), x[1:])
    edge = abs(rho) * np.abs(x).mean()
    return {"sym": sym, "tf": tf, "slice": tag, "n": len(x), "rho1": round(rho, 4), "t_rho1": round(t, 2),
            "VR2": round(vr2, 3), "z2": round(z2, 2), "VR5": round(vr5, 3), "z5": round(z5, 2), "VR10": round(vr10, 3),
            "z10": round(z10, 2), "MI_excess_x1e4": round((mi - mi0) * 1e4, 2), "E|r|_bps": round(np.abs(x).mean(), 2),
            "edge_bps": round(edge, 3), "cost_bps": COST[sym], "edge/cost": round(edge / COST[sym], 3)}


def rets(sym, tf):
    if tf == "1D":
        b = Q.bars(sym, "1D"); return np.log(b.close).diff().dropna()
    return Q.log_returns(sym, tf, intraday_only=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hurst = [], []
    for sym in Q.SYMS:
        for tf in TFS:
            r = rets(sym, tf)
            rows.append(row(sym, tf, r))
            hurst.append({"sym": sym, "tf": tf, "H_dfa": round(D.dfa_hurst(r.to_numpy()), 3)})
            if tf in ("5min", "15min", "1h"):
                hr = r.index.hour * 60 + r.index.minute
                for tag, m in (("asia", (hr >= 65) & (hr < 600)), ("europe", (hr >= 600) & (hr < 990)), ("us", (hr >= 990) & (hr < 1380))):
                    rows.append(row(sym, tf, r[m], tag))
                rv = Q.realized(sym).rv
                reg = rv.shift(1).rolling(250, min_periods=60).apply(lambda v: (v[-1] > np.nanquantile(v, 2 / 3)) * 2 + (v[-1] > np.nanquantile(v, 1 / 3)) * 1 - 0, raw=True)
                lab_day = reg.reindex(r.index.normalize()).to_numpy()
                for k, tag in ((0, "vol_low"), (1, "vol_mid"), (3, "vol_high")):
                    rows.append(row(sym, tf, r[lab_day == k], tag))
            print(sym, tf, flush=True)
    T = pd.DataFrame([x for x in rows if x]); H = pd.DataFrame(hurst)
    T.to_csv(OUT / "map.csv", index=False); H.to_csv(OUT / "hurst.csv", index=False)
    # cross-asset lead-lag and transfer entropy
    pairs = [("XAGUSD", "XAUUSD"), ("XAUUSD", "XAGUSD"), ("SP500", "NAS100"), ("NAS100", "SP500"), ("NAS100", "DJ30"),
             ("DJ30", "NAS100"), ("EURUSD", "XAUUSD"), ("USDJPY", "XAUUSD"), ("NAS100", "XAUUSD"), ("SP500", "GER40"),
             ("GER40", "SP500"), ("XAUUSD", "EURUSD")]
    LL = []
    for tf in ("1min", "5min"):
        P = Q.panel(sorted({s for p in pairs for s in p}), tf) * 1e4
        for a, b in pairs:
            ll = D.lead_lag(P[a], P[b], 3)
            te_ab = D.transfer_entropy(P[a].to_numpy(), P[b].to_numpy()); te_ba = D.transfer_entropy(P[b].to_numpy(), P[a].to_numpy())
            LL.append({"tf": tf, "leader": a, "follower": b, **{f"c{k:+d}": round(v, 4) for k, v in ll.items()},
                       "TE_ab_x1e4": round(te_ab * 1e4, 2), "TE_ba_x1e4": round(te_ba * 1e4, 2), "n": len(P)})
    LLt = pd.DataFrame(LL); LLt.to_csv(OUT / "leadlag.csv", index=False)
    pd.set_option("display.width", 250)
    a = T[T.slice == "all"]
    print(a[["sym", "tf", "n", "rho1", "t_rho1", "VR5", "z5", "VR10", "z10", "MI_excess_x1e4", "edge_bps", "cost_bps", "edge/cost"]].to_string(index=False))
    print("\nHurst (DFA):\n", H.pivot(index="sym", columns="tf", values="H_dfa")[TFS].to_string())
    s = T[(T.slice != "all") & (T.t_rho1.abs() > 4)].sort_values("edge/cost", ascending=False)
    print("\nslices with |t_rho1| > 4 (by edge/cost):\n", s[["sym", "tf", "slice", "n", "rho1", "t_rho1", "z10", "edge_bps", "cost_bps", "edge/cost"]].head(25).to_string(index=False))
    print("\nlead-lag:\n", LLt.to_string(index=False))
    # heatmap of z(VR10) for the full sample
    Z = a.pivot(index="sym", columns="tf", values="z10")[TFS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(Z.to_numpy(), cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(len(TFS))); ax.set_xticklabels(TFS); ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("QM-001 variance-ratio z*(q=10), DEV 2018-24: red = momentum, blue = mean reversion", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(OUT / "vr_heatmap.png", dpi=110); plt.close(fig)

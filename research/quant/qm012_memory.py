"""QM-012: is local memory persistent? (pre-condition for Hurst / VR regime switching)
Non-overlapping windows of W bars; in each window the variance ratio VR(q) and the lag-1 autocorrelation are
computed; the question is whether the past window's value predicts the next window's value (Spearman correlation
across consecutive windows) and whether a switching rule (trend-follow if past VR > 1, fade if < 1) earns anything
in the next window. DEV 2018-2024, 8 symbols, bar sizes 5 min (W = 1 week of bars), 1 h (W = 1 month), 1 d (W = 1 quarter)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q, dependence as D  # noqa: E402
from research.quant.qm001_map import COST  # noqa: E402

OUT = lab.REPORTS / "quant" / "QM-012"
CFG = {"5min": (1200, 6), "1h": (440, 4), "1D": (63, 5)}          # (window bars, q)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in Q.SYMS:
        for tf, (W, q) in CFG.items():
            r = (np.log(Q.bars(sym, "1D").close).diff().dropna() if tf == "1D" else Q.log_returns(sym, tf, intraday_only=True)).to_numpy() * 1e4
            k = len(r) // W
            vr = np.array([D.variance_ratio(r[i * W:(i + 1) * W], q)[0] for i in range(k)])
            ac = np.array([D.autocorr(r[i * W:(i + 1) * W], 1)[0] for i in range(k)])
            rho_vr = stats.spearmanr(vr[:-1], vr[1:])[0]; rho_ac = stats.spearmanr(ac[:-1], ac[1:])[0]
            # switching rule in the next window: position_t = sign(r_{t-1}) * (+1 if past VR>1 else -1)
            pnl = []
            for i in range(1, k):
                w = r[i * W:(i + 1) * W]
                s = 1.0 if vr[i - 1] > 1 else -1.0
                p = s * np.sign(w[:-1]) * w[1:]
                pnl.append(p.mean() - COST[sym] * np.mean(np.abs(np.diff(np.sign(w[:-1])))) / 2)
            pnl = np.array(pnl)
            rows.append({"sym": sym, "tf": tf, "windows": k, "spearman_VR(next,past)": round(rho_vr, 3),
                         "spearman_AC1(next,past)": round(rho_ac, 3), "p_VR": round(stats.spearmanr(vr[:-1], vr[1:])[1], 3),
                         "switch_bps_per_bar_gross+cost": round(pnl.mean(), 3), "switch_t": round(pnl.mean() / pnl.std() * np.sqrt(len(pnl)), 2)})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))

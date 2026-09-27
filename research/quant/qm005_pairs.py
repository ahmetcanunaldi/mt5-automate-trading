"""QM-005: cointegration / OU statistical arbitrage on DEV 2018-2024 (market neutral, independent of the drift).
Pairs: NAS100-SP500, DJ30-SP500, NAS100-DJ30, GER40-SP500, XAUUSD-XAGUSD; bars H1 and D1 (log prices).
A) inference: rolling Engle-Granger (share of windows with ADF p < 0.05) and OU half-life of the residual
B) causal trading: Kalman dynamic hedge (alpha_t, beta_t) -> standardized innovation z_t; OU fitted on the trailing
   window of z; Bertram-optimal band a* (cost converted to z units); enter at |z| >= a*, exit at 0 (variant X0) or at
   the opposite band (variant XB); max hold 5 half-lives. P&L per trade in bps of the y-leg notional:
   sum over the hold of (r_y - beta r_x) * side - cost (cost_y + |beta| cost_x).
Validation: t, years positive, placebo = stationary block bootstrap of the spread increments (destroys mean
reversion, keeps volatility), deflated Sharpe with the number of variants tried."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import coint, data as Q, ou, validate as VAL  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

COST = {"XAUUSD": 1.5, "XAGUSD": 14.6, "NAS100": 1.3, "DJ30": 1.2, "SP500": 1.2, "GER40": 1.3}
PAIRS = [("NAS100", "SP500"), ("DJ30", "SP500"), ("NAS100", "DJ30"), ("GER40", "SP500"), ("XAUUSD", "XAGUSD")]
CFG = {"1h": dict(win=1000, delta=1e-6), "1D": dict(win=250, delta=1e-5)}
OUT = lab.REPORTS / "quant" / "QM-005"


def trade(ly, lx, win, delta, cy, cx, exit_mode):
    k = coint.kalman_hedge(ly.to_numpy(), lx.to_numpy(), delta=delta, r_var=np.var(np.diff(ly.to_numpy())) * 0.1)
    z = (k.spread / np.sqrt(k.s2)).to_numpy()
    beta = k.beta.to_numpy()
    ry, rx = np.diff(ly.to_numpy(), prepend=np.nan), np.diff(lx.to_numpy(), prepend=np.nan)
    sd_spread = pd.Series(k.spread).rolling(win).std().to_numpy()       # spread sd in log units (for cost in z)
    trades = []
    pos, entry_i, side, b_fix, pnl, hl_bars, a_star = 0, 0, 0, 0.0, 0.0, 0, 0.0
    for i in range(win + 50, len(z) - 1):
        if pos == 0 and i % 20 == 0 or a_star == 0.0:
            p = ou.fit_ou(z[i - win:i])
            if 0 < p["half_life"] < win / 2:
                cost_z = (cy + cx * abs(beta[i])) * 1e-4 / max(sd_spread[i], 1e-9) * p["sigma_eq"]
                try:
                    a_star = ou.bertram_threshold(p["theta"], p["sigma"], cost_z)["a"]
                except Exception:
                    a_star = 0.0
                hl_bars = int(p["half_life"])
            else:
                a_star = 0.0
        if pos != 0:
            pnl += side * (ry[i] - b_fix * rx[i])
            held = i - entry_i
            done = (exit_mode == "X0" and side * z[i] >= 0) or (exit_mode == "XB" and side * z[i] >= a_star) or held >= 5 * max(hl_bars, 1)
            if done:
                trades.append({"i": i, "bps": pnl * 1e4 - (cy + cx * abs(b_fix)), "held": held})
                pos = 0
        if pos == 0 and a_star > 0 and abs(z[i]) >= a_star and np.isfinite(beta[i]):
            pos, side, entry_i, b_fix, pnl = 1, -int(np.sign(z[i])), i, beta[i], 0.0
    return pd.DataFrame(trades), z


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    rows_a, rows_b, series = [], [], {}
    n_variants = len(PAIRS) * len(CFG) * 2
    for tf, cfg in CFG.items():
        for y, x in PAIRS:
            P = np.log(pd.concat({y: Q.bars(y, tf).close, x: Q.bars(x, tf).close}, axis=1, join="inner").dropna())
            # A) inference
            ps, hls = [], []
            for s in range(0, len(P) - cfg["win"], cfg["win"] // 4):
                w = P.iloc[s:s + cfg["win"]]
                eg = coint.engle_granger(w[y].to_numpy(), w[x].to_numpy())
                ps.append(eg["adf_p"]); hls.append(ou.fit_ou(eg["resid"])["half_life"])
            rows_a.append({"tf": tf, "pair": f"{y}-{x}", "windows": len(ps), "share_adf_p<0.05": round(np.mean(np.array(ps) < 0.05), 2),
                           "median_half_life_bars": round(float(np.median(hls)), 1)})
            # B) trading
            for mode in ("X0", "XB"):
                T, z = trade(P[y], P[x], cfg["win"], cfg["delta"], COST[y], COST[x], mode)
                if len(T) < 10:
                    rows_b.append({"tf": tf, "pair": f"{y}-{x}", "exit": mode, "trades": len(T)}); continue
                T["time"] = P.index[T.i.to_numpy()]
                yr = T.groupby(T.time.dt.year).bps.sum()
                row = {"tf": tf, "pair": f"{y}-{x}", "exit": mode, "trades": len(T), "per_yr": round(len(T) / 7, 1),
                       "bps/trade": round(T.bps.mean(), 2), "t": round(T.bps.mean() / T.bps.std() * np.sqrt(len(T)), 2),
                       "win%": round((T.bps > 0).mean(), 2), "held_bars": round(T.held.mean(), 1),
                       "yrs_pos": f"{(yr > 0).sum()}/{len(yr)}",
                       "DSR": round(VAL.deflated_sharpe(T.bps.to_numpy(), n_variants), 3)}
                # placebo: rebuild y as x-hedged random walk with bootstrapped spread increments (no mean reversion)
                if T.bps.mean() > 0:
                    rng = np.random.default_rng(0); null = []
                    sp = (P[y] - P[x]).diff().dropna().to_numpy()
                    for kk in range(30):
                        inc = sp[VAL.stationary_bootstrap(len(sp), 50, rng)]
                        ly_null = P[x].iloc[1:] + np.cumsum(inc) + (P[y].iloc[0] - P[x].iloc[0])
                        Tn, _ = trade(pd.Series(np.r_[P[y].iloc[0], ly_null.to_numpy()], index=P.index), P[x], cfg["win"], cfg["delta"], COST[y], COST[x], mode)
                        null.append(Tn.bps.mean() if len(Tn) else 0.0)
                    row.update({"placebo_mean": round(float(np.mean(null)), 2), "placebo_p": round(float(np.mean(np.array(null) >= T.bps.mean())), 2)})
                rows_b.append(row); print(row, flush=True)
                led.log("QM-005", "pairs_ou", {"tf": tf, "pair": f"{y}-{x}", "exit": mode}, {"bps": row["bps/trade"], "t": row["t"]})
    A = pd.DataFrame(rows_a); Bt = pd.DataFrame(rows_b)
    A.to_csv(OUT / "cointegration.csv", index=False); Bt.to_csv(OUT / "trading.csv", index=False)
    pd.set_option("display.width", 230)
    print(A.to_string(index=False)); print(Bt.to_string(index=False))

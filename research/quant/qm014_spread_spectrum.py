"""QM-014: GNSS-style acquisition on market returns ("is there a strong signal hidden under the noise?").

GNSS analogy
  * code phase  -> start slot b of the server day (5-min slots, 288 per day)
  * Doppler bin -> holding horizon h (1 .. 48 slots = 5 min .. 4 h)
  * correlator  -> X_d(b, h) = sum of (whitened) 5-min log returns of day d over slots b .. b+h-1
  * coherent integration over days -> t(b, h) = mean_d X / se  (the cross-ambiguity function, CAF)
  * CFAR threshold -> 95 % quantile of max |t| over the whole grid under day-level random sign flips
    (keeps every intraday dependence and heteroskedasticity, removes only a consistent sign = the "code")
  * acquisition 2018-2021, tracking (out of sample) 2022-2024; DEV data only (lockbox untouched)
  * coherent vs non-coherent integration over years: sum_y t_y (phase-stable signal) vs sum_y t_y^2 (energy with
    unknown / flipping phase, like integrating across navigation-bit flips)
  * processing-gain ledger of the live book (v2.2 MT5 trade history): per-trade SNR, trades needed, cost in R.
Whitening = divide each day's returns by the ex-ante (lagged EWMA) daily realized volatility, the finance analogue
of pre-whitening / pulse blanking against a jammer (crisis days)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q  # noqa: E402
from research.quant.qm001_map import COST  # noqa: E402

OUT = lab.REPORTS / "quant" / "QM-014"
SYMS = sys.argv[1:] or ["XAUUSD", "NAS100", "DJ30", "EURUSD"]
FX = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
FX_COMMISSION = 5.0          # USD per lot round turn (FundingPips FX / metals)
FX_SLIP_PTS = 3.0            # per side, as research.symbols.COSTS["EURUSD"]
H = [1, 3, 6, 12, 24, 48]
NS = 288
ACQ, TRK = ("2018-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31")
B = 1000
TAG = ""
rng = np.random.default_rng(14)


def day_matrix(sym, mid=True):
    """days x 288 matrix of 5-min log returns (bp); flat slots = 0; mask of slots that had a bar.
    mid=True uses the mid price (bid close + half the bar's average spread): the bid series carries the broker's
    rollover spread schedule, a deterministic 'code' of the quote process, not of the market."""
    b = Q.bars(sym, "5min")
    pt = __import__("research.symbols", fromlist=["SPECS"]).SPECS[sym]["point"]
    lc = np.log(b.close + (b.spread.fillna(0) * pt / 2 if mid else 0))
    day = b.index.normalize()
    slot = (b.index.hour * 60 + b.index.minute) // 5
    days = day.unique()
    P = np.full((len(days), NS), np.nan)
    di = np.searchsorted(days, day)
    P[di, slot] = lc.to_numpy()
    have = ~np.isnan(P)
    P = pd.DataFrame(P).ffill(axis=1).to_numpy()
    R = np.nan_to_num(np.diff(P, axis=1, prepend=np.nan)) * 1e4
    rv = pd.Series((R ** 2).sum(1), index=days)
    sig = np.sqrt(rv.ewm(halflife=10).mean().shift(1)).to_numpy()
    return days, R, have, sig


def commission_bp(sym):
    """round-turn commission alone in bp: a hard floor that no spread improvement removes."""
    if sym not in FX:
        return 0.0
    px = Q.bars(sym, "1D").close.median()
    return FX_COMMISSION / (100_000.0 * (px if sym[3:] == "USD" else 1.0)) * 1e4


def slot_cost_bp(sym, slip=True):
    """round-trip cost in bp per 5-min slot of the server day: median recorded spread at that slot (DEV) +
    slippage both sides + commission. Non-FX symbols: the flat QM-001 cost."""
    if sym not in FX:
        return np.full(NS, COST[sym])
    b = Q.bars(sym, "5min")
    pt = __import__("research.symbols", fromlist=["SPECS"]).SPECS[sym]["point"]
    sp = (b.spread * pt / b.close * 1e4).groupby((b.index.hour * 60 + b.index.minute) // 5).median()
    sp = sp.reindex(range(NS)).ffill().bfill().to_numpy()
    px = b.close.median()
    notional_usd = 100_000.0 * (px if sym[3:] == "USD" else 1.0)
    comm = FX_COMMISSION / notional_usd * 1e4
    slip_bp = 2 * FX_SLIP_PTS * pt / px * 1e4 if slip else 0.0
    return sp + slip_bp + comm


def caf(X, n_ok):
    n = X.shape[0]
    m = X.mean(0); s = X.std(0, ddof=1)
    t = np.where(n_ok, m / (s / np.sqrt(n) + 1e-12), 0.0)
    return t, m


def surrogate_max(X, n_ok, B=B):
    """distribution of max |t| over the grid under day-level sign flips (CFAR)."""
    n = X.shape[0]
    s2 = (X ** 2).mean(0)
    out = np.empty(B)
    for i in range(0, B, 100):
        E = rng.choice([-1.0, 1.0], size=(100, n))
        M = E @ X / n
        V = (s2 - M ** 2) * n / (n - 1)
        T = np.where(n_ok, M / np.sqrt(V / n + 1e-18), 0.0)
        out[i:i + 100] = np.abs(T).max(1)
    return out


def grid(R, have, sig, rows, whiten):
    Rw = R[rows] / sig[rows, None] * np.nanmedian(sig) if whiten else R[rows]
    C = np.concatenate([np.zeros((Rw.shape[0], 1)), np.cumsum(Rw, 1)], 1)
    Hv = have[rows]
    cells, Xs, ok = [], [], []
    for h in H:
        for b0 in range(0, NS - h + 1):
            Xs.append(C[:, b0 + h] - C[:, b0])
            cells.append((b0, h))
            ok.append(Hv[:, b0:b0 + h].any(1).mean() > 0.8)
    return np.array(cells), np.column_stack(Xs), np.array(ok)


def hhmm(slot):
    return f"{slot * 5 // 60:02d}:{slot * 5 % 60:02d}"


def analyse(sym, whiten=True, mid=True):
    days, R, have, sig = day_matrix(sym, mid)
    keep = np.isfinite(sig) & (sig > 0)
    ia = keep & (days >= ACQ[0]) & (days <= ACQ[1])
    it = keep & (days >= TRK[0]) & (days <= TRK[1])
    cells, Xa, ok = grid(R, have, sig, ia, whiten)
    _, Xt, okt = grid(R, have, sig, it, whiten)
    ok = ok & okt
    ta, ma = caf(Xa, ok); tt, mt = caf(Xt, ok)
    thr_a = np.quantile(surrogate_max(Xa, ok), 0.95)
    thr_t = np.quantile(surrogate_max(Xt, ok), 0.95)
    # tracking: profile persistence corr(t_acq, t_trk) vs sign-flip null in the tracking period
    c_real = np.corrcoef(ta[ok], tt[ok])[0, 1]
    n = Xt.shape[0]; s2 = (Xt ** 2).mean(0); cn = []
    for _ in range(200):
        e = rng.choice([-1.0, 1.0], size=n); M = e @ Xt / n
        T = M / np.sqrt((s2 - M ** 2) / (n - 1) + 1e-18)
        cn.append(np.corrcoef(ta[ok], T[ok])[0, 1])
    p_track = float((np.array(cn) >= c_real).mean())
    # top acquisition cells, traded out of sample with the acquired sign, raw bp vs round-trip cost
    _, Xa_raw, _ = grid(R, have, sig, ia, False); _, Xt_raw, _ = grid(R, have, sig, it, False)
    top = []
    order, used = [], np.zeros(NS, bool)
    for k in np.argsort(-np.abs(np.where(ok, ta, 0))):          # strongest non-overlapping windows
        b0, h = cells[k]
        if not used[b0:b0 + h].any():
            order.append(k); used[b0:b0 + h] = True
        if len(order) == 6:
            break
    for k in order:
        sgn = np.sign(ta[k]); oos = sgn * Xt_raw[:, k]
        top.append({"start": hhmm(cells[k, 0]), "min": int(cells[k, 1] * 5), "t_acq": round(ta[k], 2),
                    "acq_bp": round(Xa_raw[:, k].mean(), 2), "oos_bp": round(oos.mean(), 2),
                    "oos_t": round(oos.mean() / (oos.std(ddof=1) / np.sqrt(len(oos))), 2),
                    "oos_net_bp": round(oos.mean() - slot_cost_bp(sym)[cells[k, 0]], 2)})
    # coherent vs non-coherent integration across the 7 DEV years
    yrs = pd.DatetimeIndex(days).year.to_numpy()
    cells_all, Xall, _ = grid(R, have, sig, keep, whiten)
    yk = yrs[keep]
    def yearly_t(X):
        return np.array([caf(X[yk == y], ok)[0] for y in np.unique(yk)])
    Ty = yearly_t(Xall)
    ny = Ty.shape[0]
    coh = (Ty.sum(0) / np.sqrt(ny))[ok]; ncoh = ((Ty ** 2).sum(0) - ny)[ok]
    coh_null, ncoh_null, yy_null = [], [], []
    for _ in range(100):
        e = rng.choice([-1.0, 1.0], size=Xall.shape[0])
        Tn = yearly_t(Xall * e[:, None])
        coh_null.append(np.abs(Tn.sum(0) / np.sqrt(ny))[ok].mean())
        ncoh_null.append(((Tn ** 2).sum(0) - ny)[ok].mean())
        yy_null.append(np.mean([np.corrcoef(Tn[i][ok], Tn[i + 1][ok])[0, 1] for i in range(ny - 1)]))
    yy = np.mean([np.corrcoef(Ty[i][ok], Ty[i + 1][ok])[0, 1] for i in range(ny - 1)])
    res = {"sym": sym, "cells": int(ok.sum()), "days_acq": int(ia.sum()), "days_trk": int(it.sum()),
           "max_t_acq": round(float(np.abs(ta[ok]).max()), 2), "cfar_thr_acq": round(float(thr_a), 2),
           "detections_acq": int((np.abs(ta[ok]) > thr_a).sum()),
           "max_t_trk": round(float(np.abs(tt[ok]).max()), 2), "cfar_thr_trk": round(float(thr_t), 2),
           "detections_trk": int((np.abs(tt[ok]) > thr_t).sum()),
           "corr_t_acq_trk": round(float(c_real), 3), "p_tracking": p_track,
           "coherent_mean_abs_t": round(float(np.abs(coh).mean()), 3), "coherent_null": round(float(np.mean(coh_null)), 3),
           "p_coherent": float((np.array(coh_null) >= np.abs(coh).mean()).mean()),
           "noncoh_mean_excess": round(float(ncoh.mean()), 3), "noncoh_null": round(float(np.mean(ncoh_null)), 3),
           "p_noncoherent": float((np.array(ncoh_null) >= ncoh.mean()).mean()),
           "year_to_year_corr": round(float(yy), 3), "year_to_year_null": round(float(np.mean(yy_null)), 3),
           "p_year_to_year": float((np.array(yy_null) >= yy).mean())}
    return res, top, (cells, ta, tt, ok, thr_a, thr_t)


def plot_caf(maps):
    fig, axes = plt.subplots(len(maps), 2, figsize=(12, 2.3 * len(maps)), sharex=True)
    for i, (sym, (cells, ta, tt, ok, thr_a, thr_t)) in enumerate(maps.items()):
        for j, (t, thr, lab_) in enumerate(((ta, thr_a, "acquisition 2018-21"), (tt, thr_t, "tracking 2022-24"))):
            M = np.full((len(H), NS), np.nan)
            for (b0, h), v, o in zip(cells, t, ok):
                if o:
                    M[H.index(h), b0] = v
            ax = axes[i, j]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest")
            ax.set_yticks(range(len(H))); ax.set_yticklabels([f"{h * 5}m" for h in H], fontsize=7)
            ax.set_xticks(range(0, NS, 36)); ax.set_xticklabels([hhmm(s) for s in range(0, NS, 36)], fontsize=7)
            ax.set_title(f"{sym} — {lab_} — CFAR |t| > {thr:.2f}, max |t| {np.nanmax(np.abs(M)):.2f}", fontsize=8, loc="left")
    fig.colorbar(im, ax=axes, shrink=0.6, label="t (whitened correlator output)")
    fig.suptitle("QM-014 cross-ambiguity map: start time (code phase, server time) × horizon (Doppler)", fontsize=10)
    fig.savefig(OUT / f"caf_maps{TAG}.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def matched_filter(sym, thr=(1.5, 2.0, 2.5, 3.0)):
    """Whole-code matched filter: every 5-min slot whose acquisition |t| > thr is traded in the tracking period with
    the acquired sign (each slot = one round trip). Gross vs net bp per day."""
    days, R, have, sig = day_matrix(sym, True)
    keep = np.isfinite(sig) & (sig > 0)
    ia = keep & (days >= ACQ[0]) & (days <= ACQ[1]); it = keep & (days >= TRK[0]) & (days <= TRK[1])
    Ra, Rt = R[ia], R[it]
    ta = Ra.mean(0) / (Ra.std(0, ddof=1) / np.sqrt(len(Ra)) + 1e-12)
    c_opt, c_full = slot_cost_bp(sym, slip=False), slot_cost_bp(sym, slip=True)
    out = []
    for th in thr:
        w = np.where(np.abs(ta) > th, np.sign(ta), 0.0)
        pnl = Rt @ w
        k = int((w != 0).sum())
        starts = np.flatnonzero((w != 0) & (np.r_[0.0, w[:-1]] != w))      # contiguous same-sign slots = 1 trade
        co, cf = c_opt[starts].sum(), c_full[starts].sum()
        out.append({"sym": sym, "thr": th, "slots": k, "trades_day": len(starts), "gross_bp_day": round(pnl.mean(), 2),
                    "t_gross": round(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))), 2),
                    "cost_opt_bp_day": round(co, 2), "cost_bp_day": round(cf, 2),
                    "net_opt_bp_day": round(pnl.mean() - co, 2), "net_bp_day": round(pnl.mean() - cf, 2),
                    "gross/cost_opt": round(pnl.mean() / co, 3) if co > 0 else None,
                    "breakeven_bp_trade": round(pnl.mean() / max(len(starts), 1), 3),
                    "commission_bp": round(commission_bp(sym), 3)})
    return out


def gain_ledger():
    th = pd.read_csv(lab.ROOT / "reports" / "tester" / "v22_full" / "trade_history.csv", parse_dates=["entry_time"])
    bps = th.symbol.map({"XAUUSD": COST["XAUUSD"], "NAS100.r": COST["NAS100"], "DJ30.r": COST["DJ30"]}).fillna(1.3)
    th["cost_R"] = th.entry_price * bps * 1e-4 / th.actual_sl_dist
    th["Rnet"] = th.R - th.cost_R
    yrs = (th.entry_time.max() - th.entry_time.min()).days / 365.25
    rows = []
    for key, g in list(th.groupby(["symbol", "leg"])) + [(("ALL", "book"), th)]:
        snr = g.R.mean() / g.R.std(); snr_n = g.Rnet.mean() / g.Rnet.std()
        rows.append({"symbol": key[0], "leg": key[1], "trades": len(g), "per_year": round(len(g) / yrs, 1),
                     "mean_R": round(g.R.mean(), 3), "cost_R": round(g.cost_R.mean(), 3),
                     "snr_trade_dB": round(10 * np.log10(max(snr, 1e-9) ** 2), 1),
                     "snr_net_dB": round(10 * np.log10(max(snr_n, 1e-9) ** 2), 1) if snr_n > 0 else None,
                     "gain_per_year_dB": round(10 * np.log10(len(g) / yrs), 1),
                     "t_full": round(snr * np.sqrt(len(g)), 2), "t_net": round(snr_n * np.sqrt(len(g)), 2),
                     "trades_for_t3_net": int(np.ceil((3 / snr_n) ** 2)) if snr_n > 0 else None,
                     "years_for_t3_net": round((3 / snr_n) ** 2 / (len(g) / yrs), 1) if snr_n > 0 else None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_fx" if set(SYMS) <= set(FX) else ""
    summary, tops, maps = [], {}, {}
    for sym in SYMS:
        for wh, mid in ((True, True), (False, True), (True, False)):
            res, top, mp = analyse(sym, whiten=wh, mid=mid)
            res["whiten"] = wh; res["mid"] = mid; summary.append(res)
            if wh and mid:
                tops[sym] = top; maps[sym] = mp
            print(json.dumps(res)); print(pd.DataFrame(top).to_string())
    TAG = tag
    plot_caf(maps)
    MF = pd.DataFrame([r for sym in SYMS for r in matched_filter(sym)]); MF.to_csv(OUT / f"matched_filter{tag}.csv", index=False)
    print(MF.to_string())
    S = pd.DataFrame(summary); S.to_csv(OUT / f"acquisition{tag}.csv", index=False)
    pd.concat({k: pd.DataFrame(v) for k, v in tops.items()}).to_csv(OUT / f"top_cells{tag}.csv")
    if not tag:
        G = gain_ledger(); G.to_csv(OUT / "gain_ledger.csv", index=False)
        print(G.to_string())

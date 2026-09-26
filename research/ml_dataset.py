"""EXP-014 dataset: dense M5 decision points (tick era) with price-action, microstructure and cross-asset
features, and dynamic-barrier "which is hit first" labels computed on the M1 path.

Row = (M5 bar close, direction). Label y = 1 if TP is touched before SL within the horizon (pessimistic:
SL wins inside one M1 bar), R = realized R net of spread/commission/slippage.
Barrier methods:
  VOL    : sl = a*sigma, tp = b*sigma (sigma = ATR14 on M5)                 -> adapts to regime
  STRUCT : long sl = beyond last meso swing low (+0.25 sigma), tp = next meso resistance; clipped to sigma bounds
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

from research import calendar_news, lab  # noqa: E402
from research.features import atr, ema, htf_to_ltf  # noqa: E402
from research.structure import structure_frame  # noqa: E402

DATA = lab.DATA
CROSS = ["XAGUSD", "USDX", "EURUSD", "USDJPY", "SP500", "NAS100"]
COMM_PER_OZ = 7.0 / 100.0
SLIP = 0.05
HORIZON_MIN = 240


def m5_from_m1(m1):
    return m1.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "n": "sum",
                                    "up": "sum", "dn": "sum", "path": "sum", "spread": "mean",
                                    "spread_max": "max"}).dropna(subset=["open"])


def micro_feats(m1, idx_close):
    """Microstructure over trailing 5/15/60 minutes, sampled at M5 close times."""
    out = {}
    for w in (5, 15, 60):
        r = m1[["n", "up", "dn", "path", "spread", "spread_max"]].rolling(f"{w}min")
        s = pd.DataFrame({"n": r["n"].sum(), "up": r["up"].sum(), "dn": r["dn"].sum(), "path": r["path"].sum(),
                          "spr": r["spread"].mean(), "sprmax": r["spread_max"].max()})
        net = m1["close"] - m1["close"].shift(w)
        s["imb"] = (s["up"] - s["dn"]) / (s["up"] + s["dn"]).replace(0, np.nan)
        s["eff"] = net.abs() / s["path"].replace(0, np.nan)
        s["n_z"] = s["n"] / m1["n"].rolling("1440min").mean() / w
        s["spr_z"] = s["spr"] / m1["spread"].rolling("1440min").mean()
        s["sprmax_r"] = s["sprmax"] / s["spr"]
        s.index = s.index + pd.Timedelta(minutes=1)          # M1 bar t is known at t+1min
        s = s.reindex(idx_close, method="ffill")
        for col in ("imb", "eff", "n_z", "spr_z", "sprmax_r"):
            out[f"mi{w}_{col}"] = s[col].to_numpy()
    return pd.DataFrame(out, index=idx_close)


def cross_feats(idx_close, gold_m1):
    out = {}
    g = gold_m1["close"]
    g.index = g.index + pd.Timedelta(minutes=1)
    gret = {}
    for w in (5, 15, 60):
        gr = np.log(g).diff(w)
        gret[w] = gr.reindex(idx_close, method="ffill")
    for sym in CROSS:
        p = DATA / f"{sym}_M1T.parquet"
        if not p.exists():
            continue
        c = pd.read_parquet(p)["close"]
        c.index = c.index + pd.Timedelta(minutes=1)
        lr = np.log(c)
        sd5 = lr.diff(5).rolling(2880, min_periods=500).std()
        for w in (5, 15, 60):
            r = (lr.diff(w) / (sd5 * np.sqrt(w / 5))).reindex(idx_close, method="ffill")
            out[f"x_{sym}_r{w}"] = r.to_numpy()
        if sym == "XAGUSD":
            # gold residual vs silver over 15/60 min (rolling beta on 5-min returns)
            s5 = lr.diff(5).reindex(idx_close, method="ffill")
            g5 = gret[5]
            beta = (g5.rolling(576).cov(s5) / s5.rolling(576).var()).clip(0, 2)
            for w in (15, 60):
                sw = lr.diff(w).reindex(idx_close, method="ffill")
                out[f"x_gold_resid{w}"] = ((gret[w] - beta * sw) / (g5.rolling(576).std() * np.sqrt(w / 5))).to_numpy()
    return pd.DataFrame(out, index=idx_close)


@njit(cache=True)
def first_hit(entry_i, n_i, d, sl, tp, o, h, l, c, spr, day):
    """Walk the M1 path from entry_i (entry at open). Returns (label, R)."""
    out_y = np.zeros(len(entry_i)); out_r = np.full(len(entry_i), np.nan); out_o = np.zeros(len(entry_i))
    for k in range(len(entry_i)):
        e = entry_i[k]
        if e < 0 or e >= len(o) or not (sl[k] > 0) or not (tp[k] > 0):
            continue
        cost = COMM_PER_OZ
        if d[k] == 1:
            ep = o[e] + spr[e] + SLIP
            s_px = ep - sl[k]; t_px = ep + tp[k]
        else:
            ep = o[e] - SLIP
            s_px = ep + sl[k]; t_px = ep - tp[k]
        res = np.nan
        last = min(e + n_i, len(o)) - 1
        for j in range(e, last + 1):
            if day[j] != day[e]:
                last = j - 1
                break
            if d[k] == 1:
                if l[j] <= s_px:
                    res = (s_px - SLIP - ep - cost) / sl[k]; out_o[k] = -1.0; break
                if h[j] >= t_px:
                    res = (t_px - ep - cost) / sl[k]; out_y[k] = 1.0; out_o[k] = 1.0; break
            else:
                if h[j] + spr[j] >= s_px:
                    res = (ep - s_px - SLIP - cost) / sl[k]; out_o[k] = -1.0; break
                if l[j] + spr[j] <= t_px:
                    res = (ep - t_px - cost) / sl[k]; out_y[k] = 1.0; out_o[k] = 1.0; break
        if np.isnan(res):
            j = max(last, e)
            xp = c[j] if d[k] == 1 else c[j] + spr[j]
            res = ((xp - ep) * d[k] - cost) / sl[k]
        out_r[k] = res
    return out_y, out_r, out_o


def build(save=True):
    m1 = pd.read_parquet(DATA / "XAUUSD_M1T.parquet")
    m5 = m5_from_m1(m1)
    close_t = m5.index + pd.Timedelta(minutes=5)
    a5 = atr(m5, 14)
    F = pd.DataFrame(index=m5.index)
    F["hour"] = m5.index.hour + m5.index.minute / 60
    F["dow"] = m5.index.dayofweek
    F["atr_bps"] = a5 / m5["close"] * 1e4
    F["atr_ratio"] = a5 / atr(m5, 96)
    F["atr_ratio_d"] = a5 / atr(m5, 288 * 5)
    for k in (1, 3, 6, 12, 24, 48):
        F[f"ret{k}"] = (m5["close"] - m5["close"].shift(k)) / a5
    F["clv"] = ((m5.close - m5.low) - (m5.high - m5.close)) / (m5.high - m5.low).replace(0, np.nan)
    F["body"] = (m5.close - m5.open) / a5
    for k, nm in ((2.0, "mi"), (4.0, "me"), (8.0, "ma")):
        sf, _ = structure_frame(m5, a5, k, nm)
        F = F.join(sf.drop(columns=[f"{nm}_last_h", f"{nm}_last_l", f"{nm}_prev_h", f"{nm}_prev_l"]))
        if nm == "me":
            me_last_l = sf["me_last_l"]; me_last_h = sf["me_last_h"]
    # daily levels
    date = m5.index.normalize()
    dly = m5.groupby(date).agg(dh=("high", "max"), dl=("low", "min")).shift(1)
    F["pdh_d"] = (m5["close"] - dly["dh"].reindex(date).to_numpy()) / a5
    F["pdl_d"] = (m5["close"] - dly["dl"].reindex(date).to_numpy()) / a5
    F["tdh_d"] = (m5["close"] - m5.groupby(date)["high"].cummax()) / a5
    F["tdl_d"] = (m5["close"] - m5.groupby(date)["low"].cummin()) / a5
    for step in (10, 50):
        F[f"round{step}_d"] = ((m5["close"] / step).round() * step - m5["close"]) / a5
    # H1 trend
    h1 = lab.load("H1")
    hh = pd.DataFrame({"e50": ema(h1.close, 50), "e200": ema(h1.close, 200), "a": atr(h1, 14), "c": h1.close})
    ctx = htf_to_ltf(close_t, hh, 60, ["e50", "e200", "a", "c"])
    F["h1_trend"] = ((ctx["e50"] - ctx["e200"]) / ctx["a"]).to_numpy()
    F["h1_pos"] = ((ctx["c"] - ctx["e50"]) / ctx["a"]).to_numpy()
    # microstructure + cross asset (indexed by close time)
    mi = micro_feats(m1, close_t); mi.index = m5.index
    xa = cross_feats(close_t, m1); xa.index = m5.index
    F = F.join(mi).join(xa)
    # news proximity + blackout at the entry time
    news = calendar_news.load_news_server_times()
    t = close_t.values.astype("datetime64[ns]"); nn = np.sort(news.values.astype("datetime64[ns]"))
    i = np.searchsorted(nn, t)
    nxt = np.where(i < len(nn), nn[np.minimum(i, len(nn) - 1)], t + np.timedelta64(30, "D"))
    prv = np.where(i > 0, nn[np.maximum(i - 1, 0)], t - np.timedelta64(30, "D"))
    F["news_next_h"] = np.minimum((nxt - t) / np.timedelta64(1, "h"), 72)
    F["news_prev_h"] = np.minimum((t - prv) / np.timedelta64(1, "h"), 72)
    tmin = close_t.hour * 60 + close_t.minute
    ok = (tmin >= 65) & (tmin <= 1320) & (F["news_next_h"] * 60 >= 30) & (F["news_prev_h"] * 60 >= 30)
    ok &= ~((close_t.dayofweek == 4) & (tmin >= 1290))
    F = F[np.asarray(ok)]
    F["close_time"] = close_t[np.asarray(ok)]
    F["atr"] = a5.reindex(F.index)
    F["me_last_l"] = me_last_l.reindex(F.index); F["me_last_h"] = me_last_h.reindex(F.index)
    F["close"] = m5["close"].reindex(F.index)

    # rows for both directions
    F = F.copy()
    rows = []
    for d in (1, -1):
        X = F.copy()
        X["dir"] = d
        rows.append(X)
    D = pd.concat(rows)
    D = D.reset_index(names="bar_time").sort_values(["close_time", "dir"]).reset_index(drop=True)
    # barriers
    s = D["atr"].to_numpy()
    c = D["close"].to_numpy(); d = D["dir"].to_numpy()
    lab_cols = {}
    for a_, b_ in ((1.5, 2.25), (2.0, 3.0), (3.0, 4.5), (1.5, 1.5), (2.0, 2.0), (3.0, 3.0), (3.0, 6.0)):
        lab_cols[f"VOL_{a_}_{b_}"] = (a_ * s, b_ * s)
    # structure barriers
    stop_ref = np.where(d == 1, c - D["me_last_l"].to_numpy(), D["me_last_h"].to_numpy() - c)
    tgt_ref = np.where(d == 1, D["me_res_d"].to_numpy(), D["me_sup_d"].to_numpy()) * s
    sl_st = np.clip(stop_ref + 0.25 * s, 1.0 * s, 4.0 * s)
    sl_st = np.where(np.isfinite(sl_st), sl_st, 2.0 * s)
    tp_st = np.clip(np.where(np.isfinite(tgt_ref), tgt_ref - 0.1 * s, 2.0 * sl_st), 1.0 * sl_st, 4.0 * sl_st)
    lab_cols["STRUCT"] = (sl_st, tp_st)
    # M1 path
    o, h, l, cc = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = np.maximum(m1["spread"].to_numpy(float), 15) * 0.01
    day = m1.index.values.astype("datetime64[D]").astype(np.int64)
    entry_i = np.searchsorted(m1.index.values.astype("datetime64[ns]"),
                              D["close_time"].values.astype("datetime64[ns]"), side="left").astype(np.int64)
    for name, (sl, tp) in lab_cols.items():
        y, r, oc = first_hit(entry_i, HORIZON_MIN, d.astype(np.int64), sl, tp, o, h, l, cc, spr, day)
        D[f"y_{name}"] = y; D[f"R_{name}"] = r; D[f"sl_{name}"] = sl; D[f"tp_{name}"] = tp; D[f"o_{name}"] = oc
    # direction-signed features (so one model serves both sides)
    signed = [x for x in D.columns if x.startswith(("ret", "body", "clv", "h1_", "pdh", "pdl", "tdh", "tdl", "round"))
              or x.endswith(("_state", "_hh", "_hl", "_leg", "_dh", "_dl", "_imb")) or x.startswith("x_")]
    for col in signed:
        D[col] = D[col] * D["dir"]
    # ahead / behind S/R relative to direction
    for nm in ("mi", "me", "ma"):
        long_ = D["dir"] == 1
        D[f"{nm}_ahead_d"] = np.where(long_, D[f"{nm}_res_d"], D[f"{nm}_sup_d"])
        D[f"{nm}_behind_d"] = np.where(long_, D[f"{nm}_sup_d"], D[f"{nm}_res_d"])
        D[f"{nm}_ahead_t"] = np.where(long_, D[f"{nm}_res_t"], D[f"{nm}_sup_t"])
        D[f"{nm}_behind_t"] = np.where(long_, D[f"{nm}_sup_t"], D[f"{nm}_res_t"])
        D[f"{nm}_bos_with"] = np.where(long_, D[f"{nm}_bos_up"], D[f"{nm}_bos_dn"])
        D[f"{nm}_bos_against"] = np.where(long_, D[f"{nm}_bos_dn"], D[f"{nm}_bos_up"])
        D = D.drop(columns=[f"{nm}_res_d", f"{nm}_sup_d", f"{nm}_res_t", f"{nm}_sup_t", f"{nm}_bos_up", f"{nm}_bos_dn"])
    D = D.copy()
    if save:
        D.to_parquet(DATA / "ml_dataset_m5.parquet")
    return D


if __name__ == "__main__":
    D = build()
    print(D.shape)
    for c in [c for c in D.columns if c.startswith("y_")]:
        name = c[2:]
        print(f"{name:<14} P(TP first)={D[c].mean():.3f}  mean R={D['R_' + name].mean():+.3f}  "
              f"median sl$={np.nanmedian(D['sl_' + name]):.2f} tp$={np.nanmedian(D['tp_' + name]):.2f}")

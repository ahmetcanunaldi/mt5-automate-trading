"""Causal price-action structure: ATR ZigZag swings, HH/HL/LH/LL, BOS/CHoCH, swing-cluster S/R levels.

Causality: a swing high is only CONFIRMED when price has fallen `k * ATR` below the running extreme
(and vice versa). Every feature at bar i uses swings confirmed at or before bar i only.
"""
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def zigzag(h, l, atr, k):
    """Returns arrays of confirmed swings: confirm_idx, pivot_idx, price, kind (+1 high, -1 low)."""
    n = len(h)
    c_idx = np.empty(n, np.int64); p_idx = np.empty(n, np.int64)
    price = np.empty(n); kind = np.empty(n, np.int64)
    m = 0
    d = 0                      # 0 unknown, +1 looking for a high, -1 looking for a low
    ext = h[0]; ext_i = 0
    lo_ext = l[0]; lo_i = 0
    for i in range(1, n):
        thr = k * atr[i]
        if not (thr > 0):
            continue
        if d == 0:
            if h[i] > ext:
                ext = h[i]; ext_i = i
            if l[i] < lo_ext:
                lo_ext = l[i]; lo_i = i
            if ext - l[i] >= thr and ext_i < i:
                c_idx[m] = i; p_idx[m] = ext_i; price[m] = ext; kind[m] = 1; m += 1
                d = -1; lo_ext = l[i]; lo_i = i
            elif h[i] - lo_ext >= thr and lo_i < i:
                c_idx[m] = i; p_idx[m] = lo_i; price[m] = lo_ext; kind[m] = -1; m += 1
                d = 1; ext = h[i]; ext_i = i
        elif d == 1:
            if h[i] > ext:
                ext = h[i]; ext_i = i
            elif ext - l[i] >= thr:
                c_idx[m] = i; p_idx[m] = ext_i; price[m] = ext; kind[m] = 1; m += 1
                d = -1; lo_ext = l[i]; lo_i = i
        else:
            if l[i] < lo_ext:
                lo_ext = l[i]; lo_i = i
            elif h[i] - lo_ext >= thr:
                c_idx[m] = i; p_idx[m] = lo_i; price[m] = lo_ext; kind[m] = -1; m += 1
                d = 1; ext = h[i]; ext_i = i
    return c_idx[:m], p_idx[:m], price[:m], kind[:m]


@njit(cache=True)
def _structure_feats(n, c, atr, c_idx, p_idx, price, kind, sr_lookback, sr_tol):
    # outputs
    last_h = np.full(n, np.nan); last_l = np.full(n, np.nan)
    prev_h = np.full(n, np.nan); prev_l = np.full(n, np.nan)
    hh = np.zeros(n); hl = np.zeros(n)                 # last high is HH (+1) / LH (-1); last low HL (+1) / LL (-1)
    state = np.zeros(n)                                # +1 HH&HL, -1 LH&LL, 0 mixed
    bars_since = np.full(n, np.nan)
    bos_up = np.zeros(n); bos_dn = np.zeros(n)         # close beyond last confirmed swing high / low
    res_d = np.full(n, np.nan); sup_d = np.full(n, np.nan)
    res_t = np.zeros(n); sup_t = np.zeros(n)
    leg_up = np.full(n, np.nan)                        # size of last completed swing leg / atr, signed
    j = 0                                              # swings confirmed so far
    ns = len(c_idx)
    for i in range(n):
        while j < ns and c_idx[j] <= i:
            j += 1
        if j == 0:
            continue
        # last high/low among confirmed swings
        lh = np.nan; ll = np.nan; ph = np.nan; pl = np.nan; nh = 0; nl = 0
        for q in range(j - 1, -1, -1):
            if kind[q] == 1:
                if nh == 0:
                    lh = price[q]
                elif nh == 1:
                    ph = price[q]
                nh += 1
            else:
                if nl == 0:
                    ll = price[q]
                elif nl == 1:
                    pl = price[q]
                nl += 1
            if nh >= 2 and nl >= 2:
                break
        last_h[i] = lh; last_l[i] = ll; prev_h[i] = ph; prev_l[i] = pl
        if nh >= 2:
            hh[i] = 1.0 if lh > ph else -1.0
        if nl >= 2:
            hl[i] = 1.0 if ll > pl else -1.0
        if hh[i] == 1 and hl[i] == 1:
            state[i] = 1
        elif hh[i] == -1 and hl[i] == -1:
            state[i] = -1
        bars_since[i] = i - c_idx[j - 1]
        a = atr[i]
        if a > 0:
            if nh > 0 and c[i] > lh:
                bos_up[i] = (c[i] - lh) / a
            if nl > 0 and c[i] < ll:
                bos_dn[i] = (ll - c[i]) / a
            if j >= 2:
                leg_up[i] = (price[j - 1] - price[j - 2]) / a
            # S/R: nearest swing price above / below close among the last sr_lookback swings, touches within tol
            best_r = 1e18; best_s = 1e18
            q0 = max(0, j - sr_lookback)
            for q in range(q0, j):
                dd = price[q] - c[i]
                if dd > 0 and dd < best_r:
                    best_r = dd
                elif dd <= 0 and -dd < best_s:
                    best_s = -dd
            if best_r < 1e17:
                res_d[i] = best_r / a
                lvl = c[i] + best_r
                cnt = 0
                for q in range(q0, j):
                    if abs(price[q] - lvl) <= sr_tol * a:
                        cnt += 1
                res_t[i] = cnt
            if best_s < 1e17:
                sup_d[i] = best_s / a
                lvl = c[i] - best_s
                cnt = 0
                for q in range(q0, j):
                    if abs(price[q] - lvl) <= sr_tol * a:
                        cnt += 1
                sup_t[i] = cnt
    return last_h, last_l, prev_h, prev_l, hh, hl, state, bars_since, bos_up, bos_dn, res_d, sup_d, res_t, sup_t, leg_up


def structure_frame(bars: pd.DataFrame, atr: pd.Series, k: float, prefix: str, sr_lookback=40, sr_tol=0.5):
    h, l, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    a = atr.to_numpy(float)
    ci, pi, pr, kd = zigzag(h, l, a, k)
    out = _structure_feats(len(c), c, a, ci, pi, pr, kd, sr_lookback, sr_tol)
    names = ["last_h", "last_l", "prev_h", "prev_l", "hh", "hl", "state", "bars_since", "bos_up", "bos_dn",
             "res_d", "sup_d", "res_t", "sup_t", "leg"]
    f = pd.DataFrame({f"{prefix}_{nm}": v for nm, v in zip(names, out)}, index=bars.index)
    # distances to last swing points in ATR
    f[f"{prefix}_dh"] = (c - f[f"{prefix}_last_h"]) / a
    f[f"{prefix}_dl"] = (c - f[f"{prefix}_last_l"]) / a
    swings = pd.DataFrame({"confirm": bars.index[ci], "pivot": bars.index[pi], "price": pr, "kind": kd})
    return f, swings

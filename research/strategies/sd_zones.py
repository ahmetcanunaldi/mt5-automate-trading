"""Multi-timeframe supply/demand zones (base + explosive departure), traded on the first return.

Zone detection on a timeframe TF (causal: a zone exists from the CLOSE of its departure candle):
  base      : 1..max_base consecutive candles with body <= base_body * range (indecision)
  departure : the candle right after the base, range >= dep_atr * ATR(TF) and body >= 0.6 * range
  demand    : departure up   -> zone [min(base low), max(base body top)]      proximal = top, distal = bottom
  supply    : departure down -> zone [min(base body bottom), max(base high)]  proximal = bottom, distal = top
Trade: limit at the proximal line on first touch (fresh zones only), SL beyond the distal line + 0.1 ATR,
TP = rr * risk. Order valid for `max_age_h` hours. Optional trend filter: only zones in the H1 EMA50/200
trend direction.
"""
import numpy as np
import pandas as pd
from numba import njit

from research.features import atr, ema, htf_to_ltf

TF_MIN = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}


@njit(cache=True)
def _zones(o, h, l, c, a, max_base, base_body, dep_atr):
    n = len(o)
    out_i = np.empty(n, np.int64); out_d = np.empty(n, np.int64)
    out_lo = np.empty(n); out_hi = np.empty(n); out_prox = np.empty(n); out_dist = np.empty(n)
    m = 0
    for i in range(1, n):
        rng = h[i] - l[i]
        if not (a[i - 1] > 0) or rng < dep_atr * a[i - 1] or abs(c[i] - o[i]) < 0.6 * rng:
            continue
        d = 1 if c[i] > o[i] else -1
        # collect base candles immediately before the departure
        nb = 0
        lo = 1e18; hi = -1e18; btop = -1e18; bbot = 1e18
        j = i - 1
        while j >= 0 and nb < max_base:
            r = h[j] - l[j]
            if r <= 0 or abs(c[j] - o[j]) > base_body * r:
                break
            lo = min(lo, l[j]); hi = max(hi, h[j])
            btop = max(btop, max(o[j], c[j])); bbot = min(bbot, min(o[j], c[j]))
            nb += 1
            j -= 1
        if nb == 0:
            continue
        out_i[m] = i; out_d[m] = d; out_lo[m] = lo; out_hi[m] = hi
        if d == 1:
            out_prox[m] = btop; out_dist[m] = lo
        else:
            out_prox[m] = bbot; out_dist[m] = hi
        m += 1
    return out_i[:m], out_d[:m], out_lo[:m], out_hi[:m], out_prox[:m], out_dist[:m]


def resample(h1: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "H1":
        return h1
    r = {"H4": "4h", "D1": "1D"}[rule]
    return h1.resample(r).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def zone_orders(bars_tf: pd.DataFrame, tf: str, h1_ctx: pd.DataFrame, dep_atr=2.0, max_base=3, base_body=0.5,
                rr=2.0, trend="none", max_age_h=120, hold_min=720, min_risk=1.0, max_risk_atr=3.0):
    a = atr(bars_tf, 14)
    o, h, l, c = (bars_tf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    zi, zd, zlo, zhi, zprox, zdist = _zones(o, h, l, c, a.to_numpy(float), max_base, base_body, dep_atr)
    t_close = bars_tf.index[zi] + pd.Timedelta(minutes=TF_MIN[tf])
    av = a.to_numpy()[zi]
    buf = 0.1 * av
    sl = np.where(zd == 1, zdist - buf, zdist + buf)
    risk = np.abs(zprox - sl)
    tp = zprox + zd * rr * risk
    od = pd.DataFrame({"act_time": t_close, "exp_time": t_close + pd.Timedelta(hours=max_age_h), "dir": zd,
                       "entry": zprox, "sl": sl, "tp": tp, "hold_min": hold_min, "risk": risk, "atr": av,
                       "leg": f"sd_{tf}"})
    od = od[(od.risk >= min_risk) & (od.risk <= max_risk_atr * od.atr)]
    if trend == "with":
        hh = pd.DataFrame({"e50": ema(h1_ctx.close, 50), "e200": ema(h1_ctx.close, 200), "c": h1_ctx.close})
        ctx = htf_to_ltf(od["act_time"], hh, 60, ["e50", "e200", "c"])
        up = ((ctx.e50 > ctx.e200) & (ctx.c > ctx.e50)).to_numpy()
        dn = ((ctx.e50 < ctx.e200) & (ctx.c < ctx.e50)).to_numpy()
        od = od[np.where(od.dir.to_numpy() == 1, up, dn)]
    od = od.reset_index(drop=True)
    od["group"] = np.arange(len(od))
    od["stop_on_win"] = False
    return od

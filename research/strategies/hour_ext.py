"""TomTrades "hourly extension -> 50 % mean reversion" (video wsXu2Xr1nQc, "71 % win rate strategy").

Discretionary description (transcript): identify a range, wait for a clear HIGH-VOLUME extension (one-way move,
no 50 % pullback) into the extreme of the range, typically in the first ~20-40 minutes of the hourly candle,
often taking out a prior high/low; enter aggressively at the extreme (or after a small 1-min shift) with a
stop that has "breathing room" beyond the extreme; target 50 % of the extension (1:1 .. 1.5R). Trades Asia
through London, not New York.

Codified (M1 bars rebuilt from ticks; tick count = volume):
  At minute m of hour H (m_min <= m <= m_max), with hour open O and running high/low:
    up-extension   : ext = HH - O >= k * ATR_H1, close within top `near` of the hour range,
                     no pullback >= 50 % of the extension since the hour's low, hour volume so far
                     >= vol_mult x the median volume of the same minutes over the previous 20 days
    sweep (opt.)   : HH > highest high of the previous `lookback_h` hours
    range (opt.)   : efficiency of the previous `range_h` H1 closes < eff_max (rangy market)
  Entry (fade)     : "aggr"  -> next minute open
                     "shift" -> first M1 close below the prior M1 low (for a sell) within 15 minutes
  Stop             : extreme + max(stop_atr * ATR_H1, stop_ext * ext)
  Target           : extreme - tp_frac * ext   (tp_frac 0.5 = 50 % of the extension)
Mirror image for down-extensions. One trade per hour.
"""
import numpy as np
import pandas as pd

from research.features import atr


def hour_ext_orders(m1: pd.DataFrame, k=0.75, m_min=10, m_max=40, near=0.2, vol_mult=1.2, sweep=True,
                    lookback_h=4, range_filter=False, range_h=8, eff_max=0.4, entry="aggr", stop_atr=0.25,
                    stop_ext=0.25, tp_frac=0.5, sessions=((2, 15),), hold_min=120, fill_wait=3):
    idx = m1.index
    o, h, l, c = (m1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    vol = m1["n"].to_numpy(float)
    h1 = m1.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a_h1 = atr(h1, 14).shift(1)                                     # known at the hour open
    prev_hi = h1["high"].rolling(lookback_h).max().shift(1)
    prev_lo = h1["low"].rolling(lookback_h).min().shift(1)
    eff = ((h1["close"] - h1["close"].shift(range_h)).abs() /
           h1["close"].diff().abs().rolling(range_h).sum()).shift(1)
    # volume baseline: cumulative tick count by minute-of-hour, median over previous 20 days for the same hour
    minute = idx.minute.to_numpy()
    hour_start = idx.floor("1h")
    cumv = pd.Series(vol, index=idx).groupby(hour_start).cumsum()
    key = pd.DataFrame({"cv": cumv.to_numpy(), "hod": idx.hour, "mi": minute}, index=idx)
    base = key.groupby(["hod", "mi"])["cv"].transform(lambda s: s.rolling(20, min_periods=5).median().shift(1))
    base = base.to_numpy()
    cumv = cumv.to_numpy()
    hs_vals, first_pos = np.unique(hour_start.values, return_index=True)
    bounds = list(first_pos) + [len(idx)]
    rows = []
    for q in range(len(hs_vals)):
        s, e = bounds[q], bounds[q + 1]
        H = pd.Timestamp(hs_vals[q])
        hr = H.hour
        if not any(a <= hr < b for a, b in sessions) or e - s < 45:
            continue
        A = a_h1.get(H, np.nan)
        if not np.isfinite(A) or A <= 0:
            continue
        if range_filter and not (eff.get(H, np.nan) < eff_max):
            continue
        O = o[s]
        hh, ll = -np.inf, np.inf
        done = False
        for i in range(s, e):
            if h[i] > hh:
                hh = h[i]; low_after_hh = l[i]
            if l[i] < ll:
                ll = l[i]; high_after_ll = h[i]
            m = minute[i]
            if done or m < m_min or m > m_max:
                continue
            vb = base[i]
            vol_ok = np.isfinite(vb) and vb > 0 and cumv[i] >= vol_mult * vb
            rng = hh - ll
            for d in (-1, 1):                          # d = trade direction (fade)
                if d == -1:                            # fade an UP extension
                    ext = hh - O
                    ok = ext >= k * A and c[i] >= hh - near * rng and (hh - l[i]) < 0.5 * ext
                    ok = ok and (not sweep or hh > prev_hi.get(H, np.inf))
                    extreme = hh
                else:
                    ext = O - ll
                    ok = ext >= k * A and c[i] <= ll + near * rng and (h[i] - ll) < 0.5 * ext
                    ok = ok and (not sweep or ll < prev_lo.get(H, -np.inf))
                    extreme = ll
                if not (ok and vol_ok):
                    continue
                j = i
                if entry == "shift":
                    j = -1
                    for t in range(i + 1, min(i + 16, len(c))):
                        if (d == -1 and h[t] > extreme) or (d == 1 and l[t] < extreme):
                            extreme = h[t] if d == -1 else l[t]
                        if (d == -1 and c[t] < l[t - 1]) or (d == 1 and c[t] > h[t - 1]):
                            j = t; break
                    if j < 0:
                        continue
                stop_d = max(stop_atr * A, stop_ext * ext)
                sl = extreme - d * stop_d
                tp = extreme + d * tp_frac * ext
                entry_px = c[j]
                if (tp - entry_px) * d <= 0.2 or (entry_px - sl) * d <= 0.3:
                    continue
                rows.append({"act_time": idx[j] + pd.Timedelta(minutes=1),
                             "exp_time": idx[j] + pd.Timedelta(minutes=1 + fill_wait),
                             "dir": d, "entry": entry_px + (0.0 if d == -1 else 0.3), "sl": sl, "tp": tp,
                             "hold_min": hold_min, "group": int(H.value // 10**9), "stop_on_win": False,
                             "leg": "hour_ext", "ext": ext, "atr_h1": A, "minute": m})
                done = True
                break
    return pd.DataFrame(rows)

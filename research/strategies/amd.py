"""Session AMD / "Power of 3" (accumulation -> manipulation -> distribution) on M15 bars.

Times are defined in UTC and converted to server time (UTC+2 / +3 with US DST).
  model "asia_london": accumulation = Asia 00:00-07:00 UTC, manipulation window = 07:00-11:00 UTC
  model "london_ny"  : accumulation = London 07:00-12:30 UTC, manipulation window = 12:30-15:30 UTC
Rules (M15 decision bars):
  1. accumulation range [Lo, Hi]; optional tightness filter: (Hi - Lo) <= max_rng_atr * ATR_D(14)
  2. manipulation: in the window, price trades beyond ONE side by >= sweep_atr * ATR_M15
     (the first side swept; if the other side was also taken before confirmation -> no trade)
  3. confirmation: within `confirm_bars` M15 bars an M15 candle CLOSES back inside the range
  4. distribution trade against the sweep at the next bar open:
       SL = sweep extreme +/- buf_atr * ATR_M15
       TP = opposite side of the range ("range") or rr * risk ("rr") or the range midpoint ("mid")
     exit at TP/SL, max hold, news flatten or end of day. One trade per day per model.
"""
import numpy as np
import pandas as pd

from research.calendar_news import us_dst
from research.features import atr

MODELS = {"asia_london": ((0, 0), (7, 0), (11, 0)), "london_ny": ((7, 0), (12, 30), (15, 30))}


def _server(day_utc, hh, mm):
    t = pd.Timestamp(day_utc) + pd.Timedelta(hours=hh, minutes=mm)
    return t + pd.Timedelta(hours=3 if us_dst(pd.DatetimeIndex([t]))[0] else 2)


def amd_signals(m15: pd.DataFrame, model="asia_london", sweep_atr=0.25, confirm_bars=4, max_rng_atr=None,
                tp_mode="range", rr=2.0, buf_atr=0.25, hold_min=480, min_risk=1.0):
    o, h, l, c = (m15[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = m15.index
    a15 = atr(m15, 14).to_numpy()
    d1 = m15.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    atr_d = atr(d1, 14).shift(1)
    (ah, am), (mh, mm), (eh, em) = MODELS[model]
    rows = []
    for day in pd.DatetimeIndex(np.unique(idx.normalize())):
        if day.dayofweek > 4:
            continue
        t_acc0, t_man0, t_man1 = _server(day, ah, am), _server(day, mh, mm), _server(day, eh, em)
        i0, i1, i2 = idx.searchsorted(t_acc0), idx.searchsorted(t_man0), idx.searchsorted(t_man1)
        if i1 - i0 < 8 or i2 - i1 < 4:
            continue
        hi, lo = h[i0:i1].max(), l[i0:i1].min()
        ad = atr_d.get(day, np.nan)
        if max_rng_atr is not None and not (hi - lo <= max_rng_atr * ad):
            continue
        side = 0; ext = np.nan; k_sweep = -1
        for i in range(i1, min(i2 + confirm_bars, len(c))):
            a = a15[i]
            if side == 0:
                if i >= i2:
                    break
                up = h[i] >= hi + sweep_atr * a
                dn = l[i] <= lo - sweep_atr * a
                if up and dn:
                    break
                if up:
                    side, ext, k_sweep = 1, h[i], i
                elif dn:
                    side, ext, k_sweep = -1, l[i], i
                else:
                    continue
            else:
                ext = max(ext, h[i]) if side == 1 else min(ext, l[i])
                if (side == 1 and l[i] < lo) or (side == -1 and h[i] > hi):
                    break                                  # other side taken -> no clean AMD
            if i - k_sweep > confirm_bars:
                break
            inside = (c[i] < hi) if side == 1 else (c[i] > lo)
            if not inside:
                continue
            d = -side
            entry = c[i]
            sl_px = ext + side * buf_atr * a
            risk = abs(sl_px - entry)
            if tp_mode == "range":
                tp_px = lo if d == -1 else hi
            elif tp_mode == "mid":
                tp_px = (hi + lo) / 2
            else:
                tp_px = entry + d * rr * risk
            reward = (tp_px - entry) * d
            if risk < min_risk or reward <= 0.3 * risk:
                break
            rows.append({"t": idx[i] + pd.Timedelta(minutes=15), "dir": d, "sl": risk, "tp": reward,
                         "rng": hi - lo, "rng_atr_d": (hi - lo) / ad if ad > 0 else np.nan})
            break
    if not rows:
        return pd.DataFrame(columns=["dir", "sl", "tp", "hold_min", "be", "trail"])
    s = pd.DataFrame(rows).set_index("t")
    s["hold_min"] = hold_min; s["be"] = 0.0; s["trail"] = 0.0; s["leg"] = model
    return s

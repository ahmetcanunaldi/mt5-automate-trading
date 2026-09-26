"""TomTrades "CBR" Asia-reversal model (as documented publicly, e.g. fxreplay.com/strategies/tomtrades-cbr-model).

Codified rules (all on M1 bars rebuilt from ticks; server time = UTC+2/+3):
  Session  : Asia opens 00:00 UTC. First hour [00:00, 01:00) UTC defines the Asia-1 range.
             Setups are searched in the SECOND hour [01:00, 02:00) UTC.
  Bias     : H1 gold vs DXY over the last `bias_h` hours ending at 01:00 UTC. Bullish if gold up AND DXY down,
             bearish if gold down AND DXY up, otherwise skip (dxy="none": gold direction only;
             dxy="eurusd": EURUSD up counts as DXY down).
  Extension: price moves AGAINST the bias for >= `ext_min` minutes from the leg extreme with no pullback
             larger than `pb_frac` of the leg; optionally the leg must sweep the Asia-1 low (bullish) / high.
  Type-3 shift (MSB): after the extension extreme, a 1-minute candle CLOSES beyond the highest wick of the
             `k` candles that preceded the extreme (for bullish), before a new extreme is made.
  Entry    : limit at 50 % of the MSB move (extreme -> MSB high). ~30 % will not fill — accepted.
  Stop     : beyond the extension extreme (+ buffer).
  Target   : 1.5 R, or 50 % of the extension leg ("rebalance").
  Management: max 2 attempts per session, done for the day after a win (stop_on_win group).
"""
import numpy as np
import pandas as pd

from research.calendar_news import us_dst


def session_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Server-time timestamps of 00:00 UTC for every weekday in the data."""
    days = pd.DatetimeIndex(np.unique(index.normalize()))
    utc0 = days  # the UTC calendar day (00:00 UTC)
    off = np.where(us_dst(utc0 + pd.Timedelta(hours=1)), 3, 2)
    s = utc0 + pd.to_timedelta(off, unit="h")
    return s[s.dayofweek < 5]


def _h1_dir(close: pd.Series, t_end, hours):
    a = close.asof(t_end - pd.Timedelta(hours=hours))
    b = close.asof(t_end)
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0
    return int(np.sign(b - a))


def cbr_orders(m1: pd.DataFrame, dxy: pd.Series | None, bias_h=1, dxy_mode="usdx", sweep=True, tp_mode="1.5R",
               ext_min=20, pb_frac=0.35, k=5, lookback=60, msb_wait=30, fill_wait=45, buffer=0.3,
               max_attempts=2, hold_min=180, rr=1.5, min_stop=1.0):
    o, h, l, c = (m1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = m1.index
    gold_c = m1["close"]
    rows = []
    for s0 in session_starts(idx):
        t1, t2 = s0 + pd.Timedelta(hours=1), s0 + pd.Timedelta(hours=2)
        i0, i1, i2 = idx.searchsorted(s0), idx.searchsorted(t1), idx.searchsorted(t2)
        if i2 - i1 < 50 or i1 - i0 < 30:
            continue
        g = _h1_dir(gold_c, t1, bias_h)
        if dxy_mode == "none":
            bias = g
        else:
            dd = _h1_dir(dxy, t1, bias_h)
            if dxy_mode == "eurusd":
                dd = -dd
            bias = g if (g != 0 and dd == -g) else 0
        if bias == 0:
            continue
        a1_hi, a1_lo = h[i0:i1].max(), l[i0:i1].min()
        attempts = 0
        j = i1
        while j < i2 and attempts < max_attempts:
            # candidate extension extreme at bar j (against the bias)
            lo_w = max(i0, j - lookback)
            if bias == 1:
                if l[j] > l[lo_w:j].min(initial=np.inf):
                    j += 1; continue
                s = lo_w + int(np.argmax(h[lo_w:j + 1]))
                leg = h[s] - l[j]
                run_ext = np.minimum.accumulate(l[s:j + 1])
                pull = (h[s:j + 1] - run_ext).max() if j > s else 0.0
                swept = l[j] < a1_lo
            else:
                if h[j] < h[lo_w:j].max(initial=-np.inf):
                    j += 1; continue
                s = lo_w + int(np.argmin(l[lo_w:j + 1]))
                leg = h[j] - l[s]
                run_ext = np.maximum.accumulate(h[s:j + 1])
                pull = (run_ext - l[s:j + 1]).max() if j > s else 0.0
                swept = h[j] > a1_hi
            if (j - s) < ext_min or leg <= 0 or pull > pb_frac * leg or (sweep and not swept):
                j += 1; continue
            ref = h[max(s, j - k):j].max(initial=-np.inf) if bias == 1 else l[max(s, j - k):j].min(initial=np.inf)
            ext = l[j] if bias == 1 else h[j]
            msb = -1
            for m in range(j + 1, min(j + 1 + msb_wait, len(c))):
                if (bias == 1 and l[m] < ext) or (bias == -1 and h[m] > ext):
                    break                                     # new extreme -> re-evaluate from there
                if (bias == 1 and c[m] > ref) or (bias == -1 and c[m] < ref):
                    msb = m; break
            if msb < 0:
                j += 1; continue
            if bias == 1:
                top = h[j:msb + 1].max()
                entry = ext + 0.5 * (top - ext)
                sl = ext - buffer
                risk = entry - sl
                tp = entry + rr * risk if tp_mode == "1.5R" else ext + 0.5 * leg
            else:
                bot = l[j:msb + 1].min()
                entry = ext - 0.5 * (ext - bot)
                sl = ext + buffer
                risk = sl - entry
                tp = entry - rr * risk if tp_mode == "1.5R" else ext - 0.5 * leg
            if risk >= min_stop and (tp - entry) * bias > 0.3:
                rows.append({"act_time": idx[msb] + pd.Timedelta(minutes=1),
                             "exp_time": idx[msb] + pd.Timedelta(minutes=1 + fill_wait),
                             "dir": bias, "entry": entry, "sl": sl, "tp": tp, "hold_min": hold_min,
                             "group": int(s0.value // 10**9), "stop_on_win": True, "leg": "cbr",
                             "stop_usd": risk, "leg_usd": leg})
                attempts += 1
            j = msb + 1
    return pd.DataFrame(rows)

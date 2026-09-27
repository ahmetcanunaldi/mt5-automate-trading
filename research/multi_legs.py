"""Generic leg builders (same rules and parameters as the XAUUSD book) for any symbol's M1 bars.

legs_for(m1) returns {name: signals}. Calendar legs are produced in both directions (suffix _L / _S) because the
gold long bias does not carry over to FX; trend/breakout legs trade both directions natively (trendH4 as _L/_S).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.features import atr, ema  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_drift import drift_signals  # noqa: E402
from research.run_trend_intraday import scan  # noqa: E402
from research.run_volbreak import daily_ctx, signals as vb_signals  # noqa: E402
from research.strategies import swing  # noqa: E402
from research.strong_close import signals as sc_signals  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
INTRADAY = {"tday", "tday900", "lw", "inside", "nr7", "friday", "strong_close", "weak_close", "drift", "season"}


def _flip(s):
    s = s.copy(); s["dir"] = -s["dir"]; return s


def legs_for(m1):
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    h4 = m1.resample("4h").agg(AGG).dropna(subset=["open"])
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    atr_d = atr(d1, 14).shift(1)
    days = pd.DatetimeIndex(np.unique(m1.loc[A:B].index.normalize())); days = days[days.dayofweek < 5]
    L = {}
    df4 = swing.prep(h4, bar=pd.Timedelta(hours=4))
    t4 = swing.donchian(df4, n=180, sl=2.0, trail=6.0, tp=50.0, hold=240, both=True)
    L["trendH4_L"] = t4[t4.dir == 1]; L["trendH4_S"] = t4[t4.dir == -1]
    st = pd.DataFrame({"atr": atr(d1, 14), "c": d1.close, "e20": ema(d1.close, 20), "e50": ema(d1.close, 50)})
    state = np.where((st.c > st.e50) & (st.e20 > st.e50), 1, np.where((st.c < st.e50) & (st.e20 < st.e50), -1, 0))
    day = pd.DataFrame({"state": pd.Series(state, index=d1.index).shift(1), "atr": st.atr.shift(1)})
    m15ab = m15.loc[A:B]
    s = scan(m15ab, day, "tday", 0.3, 1.0, None, 1080); s["hold_min"] = 330; L["tday"] = s
    s = scan(m15ab, day, "tday", 0.3, 1.0, None, 900); s["hold_min"] = 510; L["tday900"] = s
    dr = drift_signals(m1, atr(h1, 14), entry_min=75, hold=475, days=(1, 2, 3, 4), sl_k=4.0)
    L["drift_L"] = dr; L["drift_S"] = _flip(dr)
    fri = days[days.dayofweek == 4]
    f = pd.DataFrame({"dir": 1, "sl": 1.5 * atr_d.reindex(fri).to_numpy(), "tp": 1e6, "hold_min": 1315, "be": 0.0,
                      "trail": 0.0}, index=fri + pd.Timedelta(minutes=65)).dropna()
    L["friday_L"] = f; L["friday_S"] = _flip(f)
    first = days[pd.Series(days, index=days).groupby(days.to_period("M")).cumcount().to_numpy() == 0]
    tom = pd.DataFrame({"dir": 1, "sl": 2.0 * atr_d.reindex(first).to_numpy(), "tp": 1e6, "hold_min": 3 * 1440 - 60,
                        "be": 0.0, "trail": 0.0}, index=first + pd.Timedelta(minutes=70)).dropna()
    L["tom_L"] = tom; L["tom_S"] = _flip(tom)
    ctx = daily_ctx(m1)
    L["lw"] = vb_signals(m15ab, ctx, "lw", k=0.4, sl=1.0, start_min=600)
    L["inside"] = vb_signals(m15ab, ctx, "nr", pattern="inside", trend=True)
    L["nr7"] = vb_signals(m15ab, ctx, "nr", pattern="nr7", trend=True)
    a15 = atr(m15, 14)
    fc = days[days.dayofweek == 4] + pd.Timedelta(minutes=1385)
    fcs = pd.DataFrame({"dir": 1, "sl": 4.0 * a15.asof(fc - pd.Timedelta(minutes=15)).to_numpy(), "tp": 1e6,
                        "hold_min": 50, "be": 0.0, "trail": 0.0}, index=fc).dropna()
    L["fri_close_L"] = fcs; L["fri_close_S"] = _flip(fcs)
    L["strong_close"] = sc_signals(d1, 0.6, 1.0)
    # mirror: weak close (clv < -0.6) -> short next day
    inv = d1.copy(); inv["high"], inv["low"] = -d1.low, -d1.high; inv["close"], inv["open"] = -d1.close, -d1.open
    wc = sc_signals(inv, 0.6, 1.0); wc["dir"] = -1
    wc["sl"] = atr(d1, 14).reindex(d1.index).shift(0).loc[d1.index[d1.index.searchsorted(wc.index.normalize()) - 1]].to_numpy()
    L["weak_close"] = wc
    for k in list(L):
        v = L[k].loc[A:B].copy()
        v["leg"] = k
        if k.split("_")[0] in INTRADAY or k in INTRADAY:
            v["trail"] = 1.5 * v.sl
        L[k] = v[np.isfinite(v.sl) & (v.sl > 0)]
    return L

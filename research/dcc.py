"""EXP-106: DC Capital "DCC Trade Model" (user request; rules from the Turkish walkthrough
https://www.youtube.com/watch?v=RQpcm3aEoUk of https://www.youtube.com/watch?v=zGF0OUNrW4M), EMA version.

Rules (mechanical version):
  1H bias   : EMA9 > EMA20 on the last completed H1 bar -> longs only; EMA9 < EMA20 -> shorts only.
  5M entry  : Pull-Flip-Go - within the bias, 5M EMA9 crosses against EMA20 (pullback starts while the bias already
              points the same way), then crosses back (flip); market entry at the close of the flip bar.
  VWAP      : (mandatory) daily VWAP (server day, tick volume) below the entry price for longs / above for shorts.
  ADX       : (optional) 1H ADX(14) > 20.
  Stop      : k x ATR(14) of 1H, k = 0.9 (day trading; 1.5 / 2 = "swing"); target 1.5 R; no breakeven.
  Session   : presenter searched 10:00-19:00 (Turkey time = server time in summer); variant = whole day 01:00-22:00.
  Frequency : at most 2 entries per day and symbol, one position at a time; open trades closed by 23:30 server
              (presenter closes trades that drift into the night); weekend flat; news rule v2 (no entries +-10 min,
              flatten 10 min before) from our engine. The 2H support/resistance "confirmation" is discretionary and
              optional in the model -> not used.
Test: XAUUSD + 7 FX majors, M1 execution 2019-01 .. 2026-09, costs (spread, slippage, $7/lot), fixed $500 risk.
Variants: ADX {off, >20} x k {0.9, 1.5, 2.0} x window {10-19, 01-22} = 12. Selection honesty: pick the best variant
on 2019-2022 (pooled), report 2023-2026. Placebo / baseline: the same number of entries per day at random 5M bars of
the same window in the direction of the 1H bias (tests whether Pull-Flip-Go timing adds anything to "trade with the
H1 EMA bias"), 20 draws."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, symbols  # noqa: E402
from research.features import atr, ema  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
SYMS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
OUT = lab.REPORTS / "EXP-106"
H = pd.Timedelta(hours=1)
M5 = pd.Timedelta(minutes=5)


def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, 1)
    a = lambda x: pd.Series(x, index=df.index).ewm(alpha=1 / n, adjust=False).mean()  # noqa: E731
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi, ndi = 100 * a(pdm) / atr_n, 100 * a(ndm) / atr_n
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def features(m1):
    """5M frame with everything known at each 5M bar close (index = close time)."""
    m5 = m1.resample("5min").agg(AGG).dropna(subset=["open"])
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    hf = pd.DataFrame({"bias": np.sign(ema(h1.close, 9) - ema(h1.close, 20)), "atr": atr(h1, 14), "adx": adx(h1, 14)})
    hf.index = hf.index + H                                           # known at the H1 close
    f = pd.DataFrame(index=m5.index + M5)
    f["close"] = m5.close.to_numpy()
    f = pd.merge_asof(f, hf, left_index=True, right_index=True, direction="backward")
    f["d"] = np.sign(ema(m5.close, 9) - ema(m5.close, 20)).to_numpy()
    tp = (m5.high + m5.low + m5.close) / 3
    v = m5.tick_volume.clip(lower=1)
    day = m5.index.normalize()
    f["vwap"] = ((tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()).to_numpy()
    f["mod"] = (f.index.hour * 60 + f.index.minute).to_numpy()
    return f


def pfg_events(f):
    """Pull-Flip-Go flips: 5M EMA cross back into the bias after a pullback that started within the same bias."""
    d = f.d.to_numpy(); bias = f.bias.to_numpy()
    cross = np.r_[False, (d[1:] != d[:-1]) & (d[1:] != 0) & (d[:-1] != 0)]
    idx = np.flatnonzero(cross)
    ev = []
    for a, b in zip(idx[:-1], idx[1:]):                                # a = pullback start, b = flip
        if d[b] == bias[b] and d[a] == -bias[b] and bias[a] == bias[b] and bias[b] != 0:
            ev.append(b)
    return np.array(ev, dtype=int)


def signals(f, ev, k=0.9, adx_min=0.0, win=(600, 1140), rr=1.5):
    g = f.iloc[ev]
    dirn = g.bias.to_numpy()
    ok = ((dirn > 0) & (g.vwap < g.close)) | ((dirn < 0) & (g.vwap > g.close))
    ok &= (g["mod"] >= win[0]) & (g["mod"] < win[1]) & (g.adx > adx_min) & g.atr.notna()
    g = g[ok]
    sl = k * g.atr.to_numpy()
    s = pd.DataFrame({"dir": g.bias.astype(int).to_numpy(), "sl": sl, "tp": rr * sl,
                      "hold_min": (1410 - g["mod"]).to_numpy(), "be": 0.0, "trail": 0.0, "leg": "dcc"}, index=g.index)
    return s[(s.hold_min > 5) & (s.sl > 0)].loc[A:B]


def random_like(f, s, rng, win):
    """same entries per day, random 5M bar of the window, direction = H1 bias at that bar."""
    ff = f[(f["mod"] >= win[0]) & (f["mod"] < win[1]) & (f.bias != 0) & f.atr.notna()]
    days = ff.index.normalize()
    out = []
    per_day = s.groupby(s.index.normalize()).size()
    groups = pd.Series(np.arange(len(ff)), index=days).groupby(level=0)
    pos = {d: g.to_numpy() for d, g in groups}
    for d, n in per_day.items():
        if d not in pos:
            continue
        pick = rng.choice(pos[d], size=min(n, len(pos[d])), replace=False)
        out.append(ff.iloc[np.sort(pick)])
    g = pd.concat(out)
    k = (s.sl / f.atr.reindex(s.index)).median()
    r = pd.DataFrame({"dir": g.bias.astype(int).to_numpy(), "sl": k * g.atr.to_numpy(), "hold_min": (1410 - g["mod"]).to_numpy(),
                      "be": 0.0, "trail": 0.0, "leg": "rand"}, index=g.index)
    r["tp"] = 1.5 * r.sl
    return r[r.hold_min > 5]


def guards():
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390, flatten_min=1425,
                         fri_flatten_min=1350, max_trades_day=2, max_positions=1, max_open_risk_pct=0.5,
                         risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0)


def stats(t):
    if len(t) == 0:
        return {"n": 0}
    r = t.R
    yr = t.groupby(t.entry_time.dt.year).R.sum()
    return {"n": len(t), "win": round((r > 0).mean(), 3), "avgR": round(r.mean(), 3),
            "t": round(r.mean() / r.std() * np.sqrt(len(r)), 2) if r.std() > 0 else 0.0,
            "R_yr": round(r.sum() / 7.7, 1), "yrs_pos": f"{int((yr > 0).sum())}/{len(yr)}",
            "R_19_22": round(t[t.entry_time.dt.year <= 2022].R.sum(), 1), "R_23_26": round(t[t.entry_time.dt.year >= 2023].R.sum(), 1)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(106)
    VARIANTS = list(itertools.product((0.0, 20.0), (0.9, 1.5, 2.0), ((600, 1140), (60, 1320))))
    rows, trades, X, F, EV = [], {}, {}, {}, {}
    for sym in SYMS:
        m1 = symbols.load_m1(sym).loc["2018-10-01":B]
        x = X[sym] = symbols.prepare(sym, m1.loc[A:B])
        f = F[sym] = features(m1)
        ev = EV[sym] = pfg_events(f)
        for adx_min, k, win in VARIANTS:
            s = signals(f, ev, k, adx_min, win)
            t = engine.run(x, s, guards(), symbols.COSTS[sym]).trades
            key = f"adx{int(adx_min)}_k{k}_w{win[0] // 60}-{win[1] // 60}"
            trades[(sym, key)] = t
            rows.append({"sym": sym, "variant": key, **stats(t)})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "variants.csv", index=False)
    # pooled by variant; selection on 2019-22, evaluation 2023-26
    pool = []
    for key in T.variant.unique():
        tt = pd.concat([trades[(s_, key)] for s_ in SYMS])
        st = stats(tt)
        a, b = tt[tt.entry_time.dt.year <= 2022], tt[tt.entry_time.dt.year >= 2023]
        pool.append({"variant": key, **st, "avgR_19_22": round(a.R.mean(), 3), "avgR_23_26": round(b.R.mean(), 3),
                     "t_23_26": round(b.R.mean() / b.R.std() * np.sqrt(len(b)), 2)})
    P = pd.DataFrame(pool).sort_values("avgR_19_22", ascending=False); P.to_csv(OUT / "pooled.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nPOOLED (8 symbols) by variant, sorted by 2019-22 avg R:\n", P.to_string(index=False))
    best = P.variant.iloc[0]
    print(f"\nselected on 2019-22: {best}; its 2023-26 result is the honest out-of-sample estimate")
    # baseline: random entries in the H1-bias direction, same count per day (20 draws) for the selected variant
    adx_min, k, win = [v for v in VARIANTS if f"adx{int(v[0])}_k{v[1]}_w{v[2][0] // 60}-{v[2][1] // 60}" == best][0]
    real = pd.concat([trades[(s_, best)] for s_ in SYMS]).R.mean()
    null = []
    for draw in range(20):
        rr_ = []
        for sym in SYMS:
            s = signals(F[sym], EV[sym], k, adx_min, win)
            rr_.append(engine.run(X[sym], random_like(F[sym], s, rng, win), guards(), symbols.COSTS[sym]).trades.R)
        null.append(pd.concat(rr_).mean())
        print("draw", draw, round(null[-1], 4), flush=True)
    null = np.array(null)
    print(f"\nselected variant avg R {real:.3f} vs random-timing-with-bias baseline mean {null.mean():.3f} "
          f"(95 % {np.quantile(null, 0.95):.3f}); p = {(null >= real).mean():.2f}")
    pd.Series({"best": best, "real_avgR": real, "null_mean": null.mean(), "null_q95": np.quantile(null, 0.95),
               "p": (null >= real).mean()}).to_csv(OUT / "baseline.csv")

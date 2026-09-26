"""EXP-037: classic intraday volatility breakouts on gold (2019-2026, M15 decisions, M1 execution, $100k).

  lw    : Larry Williams — long when price exceeds day_open + k * prev_range, short below day_open - k * prev_range;
          first trigger of the day only; SL = sl * ATR_D, exit at end of day (or TP rr * risk).
  nr    : Crabel — after an NR4/NR7 (narrowest range of the last 4/7 days) or inside day, trade the first break of
          yesterday's high/low; SL = the other side of yesterday's range (capped at 1.5 ATR_D), exit end of day.
Optional trend filter: only in the direction of the D1 EMA20/50 state.
"""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr, ema  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def daily_ctx(m1):
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    rng = d1.high - d1.low
    c = pd.DataFrame(index=d1.index)
    c["prev_rng"] = rng.shift(1); c["pdh"] = d1.high.shift(1); c["pdl"] = d1.low.shift(1)
    c["atr"] = atr(d1, 14).shift(1)
    c["nr4"] = (rng == rng.rolling(4).min()).shift(1); c["nr7"] = (rng == rng.rolling(7).min()).shift(1)
    c["inside"] = ((d1.high < d1.high.shift(1)) & (d1.low > d1.low.shift(1))).shift(1)
    e20, e50 = ema(d1.close, 20), ema(d1.close, 50)
    c["state"] = pd.Series(np.where((d1.close > e50) & (e20 > e50), 1, np.where((d1.close < e50) & (e20 < e50), -1, 0)),
                           index=d1.index).shift(1)
    return c


def signals(m15, ctx, kind, k=0.3, sl=0.5, rr=None, trend=False, pattern="nr4", start_min=65, end_min=1200):
    rows = []
    for day, g in m15.groupby(m15.index.normalize()):
        if day not in ctx.index:
            continue
        cx = ctx.loc[day]
        if not (cx.atr > 0):
            continue
        tm = (g.index.hour * 60 + g.index.minute).to_numpy()
        ok = (tm >= start_min) & (tm <= end_min)
        if kind == "lw":
            o = g.open.iloc[0]
            up_lvl, dn_lvl = o + k * cx.prev_rng, o - k * cx.prev_rng
        else:
            if not bool(cx[pattern]):
                continue
            up_lvl, dn_lvl = cx.pdh, cx.pdl
        up = (g.close.to_numpy() > up_lvl) & ok
        dn = (g.close.to_numpy() < dn_lvl) & ok
        iu = np.argmax(up) if up.any() else 10**9
        idn = np.argmax(dn) if dn.any() else 10**9
        if iu == idn == 10**9:
            continue
        d, i = (1, iu) if iu < idn else (-1, idn)
        if trend and cx.state != d:
            continue
        entry = g.close.iloc[i]
        if kind == "lw":
            risk = sl * cx.atr
        else:
            risk = min(abs(entry - (cx.pdl if d == 1 else cx.pdh)), 1.5 * cx.atr)
        if risk <= 0:
            continue
        rows.append({"t": g.index[i] + pd.Timedelta(minutes=15), "dir": d, "sl": risk,
                     "tp": rr * risk if rr else 50 * cx.atr})
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows).set_index("t")
    s["hold_min"] = 1440; s["be"] = 0.0; s["trail"] = 0.0; s["leg"] = kind
    return s


if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"]).loc[A:B]
    ctx = daily_ctx(m1)
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    g = engine.Guards(initial_balance=100_000, max_trades_day=2, total_stop_pct=100.0, total_derisk_pct=100.0)
    cfgs = [("lw", dict(k=k, sl=sl, rr=rr, trend=tr, start_min=st)) for k, sl, rr, tr, st in
            itertools.product([0.2, 0.4, 0.6], [0.5, 1.0], [None, 2.0], [False, True], [65, 600])]
    cfgs += [("nr", dict(pattern=p, rr=rr, trend=tr, start_min=st)) for p, rr, tr, st in
             itertools.product(["nr4", "nr7", "inside"], [None, 2.0], [False, True], [65, 600])]
    rows, keep = [], {}
    for kind, p in cfgs:
        s = signals(m15, ctx, kind, **p)
        if len(s) < 30:
            continue
        res = engine.run(x, s, g)
        t = res.trades
        m = metrics.summarize(res, "")
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"{kind}|" + "|".join(f"{k}{v}" for k, v in p.items())
        rows.append({"cfg": name, "n": len(t), "wr": m["win_rate"], "avgR": m["avg_R"],
                     "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "R_yr": round(t.R.sum() / 7.7, 1),
                     "SR": m["sharpe"], "DD%": m["max_total_dd_pct"], "yrs_pos": int((yr > 0).sum()),
                     **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("t", ascending=False)
    (lab.REPORTS / "EXP-037").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-037" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.head(25).to_string(index=False))
    print(f"configs {len(T)}, avgR>0 {(T.avgR > 0).mean():.2f}, >=7/8 yrs {(T.yrs_pos >= 7).sum()}")
    top = list(T.cfg.head(2))
    lab.save_experiment("EXP-037", {"strategy": "vol breakouts"}, {c: keep[c][0] for c in top},
                        {c.replace("|", "_"): keep[c][1] for c in top})

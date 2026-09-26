"""EXP-043: multi-timeframe momentum alignment. Trade M15 Donchian breakouts only when the M15, H1, H4 and D1
trend states (EMA fast > slow and close > fast) all point the same way. SL 2 ATR_M15, trail 3 ATR_M15 or none,
exit at end of day (intraday) or swing (hold up to 3 days). M1 2019-2026, $100k."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr, ema, htf_to_ltf  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def state(df, f, s):
    ef, es = ema(df.close, f), ema(df.close, s)
    return pd.Series(np.where((ef > es) & (df.close > ef), 1, np.where((ef < es) & (df.close < ef), -1, 0)), index=df.index)


if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    ctx = {}
    for tf, rule, mins, f, s in (("h1", "1h", 60, 50, 200), ("h4", "4h", 240, 20, 50), ("d1", "1D", 1440, 20, 50)):
        b = m1.resample(rule).agg(AGG).dropna(subset=["open"])
        st = pd.DataFrame({"st": state(b, f, s)})
        ctx[tf] = htf_to_ltf(m15.index + pd.Timedelta(minutes=15), st, mins, ["st"])["st"].to_numpy()
    s15 = state(m15, 20, 50).to_numpy()
    a15 = atr(m15, 14)
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows, keep = [], {}
    tm = (m15.index.hour * 60 + m15.index.minute).to_numpy()
    for n, need, trail, swing_, win in itertools.product([16, 32], ["all4", "h1h4d1", "h4d1"], [0.0, 3.0], [False, True],
                                                        [(65, 1320), (600, 1260)]):
        hi = m15.high.rolling(n).max().shift(1).to_numpy(); lo = m15.low.rolling(n).min().shift(1).to_numpy()
        c = m15.close.to_numpy(); o = m15.open.to_numpy()
        if need == "all4":
            up = (s15 == 1) & (ctx["h1"] == 1) & (ctx["h4"] == 1) & (ctx["d1"] == 1)
            dn = (s15 == -1) & (ctx["h1"] == -1) & (ctx["h4"] == -1) & (ctx["d1"] == -1)
        elif need == "h1h4d1":
            up = (ctx["h1"] == 1) & (ctx["h4"] == 1) & (ctx["d1"] == 1); dn = (ctx["h1"] == -1) & (ctx["h4"] == -1) & (ctx["d1"] == -1)
        else:
            up = (ctx["h4"] == 1) & (ctx["d1"] == 1); dn = (ctx["h4"] == -1) & (ctx["d1"] == -1)
        w = (tm >= win[0]) & (tm <= win[1])
        L = w & up & (c > hi) & (o <= hi); S = w & dn & (c < lo) & (o >= lo)
        idx = m15.index
        parts = []
        for msk, d in ((L, 1), (S, -1)):
            parts.append(pd.DataFrame({"dir": d, "sl": 2.0 * a15.to_numpy()[msk], "tp": 1e6,
                                       "hold_min": 3 * 1440 if swing_ else 1440, "be": 0.0,
                                       "trail": trail * a15.to_numpy()[msk], "leg": "mtf"},
                                      index=idx[msk] + pd.Timedelta(minutes=15)))
        sig = pd.concat(parts).sort_index().loc[A:B].dropna()
        g = engine.Guards(initial_balance=100_000, intraday=not swing_, weekend_flat=True, max_trades_day=4,
                          total_stop_pct=100.0, total_derisk_pct=100.0)
        res = engine.run(x, sig, g)
        t = res.trades
        if len(t) < 30:
            continue
        m = metrics.summarize(res, "")
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"n{n}|{need}|tr{trail}|{'swing' if swing_ else 'intra'}|{win[0]}-{win[1]}"
        rows.append({"cfg": name, "n": len(t), "wr": m["win_rate"], "avgR": m["avg_R"],
                     "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "R_yr": round(t.R.sum() / 7.7, 1),
                     "SR": m["sharpe"], "DD%": m["max_total_dd_pct"], "yrs_pos": int((yr > 0).sum()),
                     **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows).sort_values("t", ascending=False)
    (lab.REPORTS / "EXP-043").mkdir(exist_ok=True)
    R.to_csv(lab.REPORTS / "EXP-043" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False))
    top = list(R.cfg.head(2))
    lab.save_experiment("EXP-043", {"strategy": "MTF momentum alignment"}, {c: keep[c][0] for c in top},
                        {c.replace("|", "_"): keep[c][1] for c in top})

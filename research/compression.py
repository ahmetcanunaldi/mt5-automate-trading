"""EXP-076: compression-breakout family (the inside-day leg has the best R per trade of the book: +0.28 R, 82 trades).
Break of the previous day's high/low after a compression pattern, D1 trend filter (EMA20/50 state), SL = other side
of yesterday's range capped at 1.5 ATR_D, exit end of day (same execution as the inside/nr7 legs).
Patterns: inside, nr7 (current), nr4, id_nr4 (inside and NR4), inside2 (two inside days in a row), nr7_long_only
without trend filter. Split-half (2019-22 / 2023-26) and portfolio impact on best13."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, legcache  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.news_scope import book_row  # noqa: E402
from research.payout_sim import A  # noqa: E402
from research.run_volbreak import daily_ctx, signals  # noqa: E402

if __name__ == "__main__":
    m1, legs = legcache.load()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"]).loc[A:B]
    ctx = daily_ctx(m1)
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    rng = d1.high - d1.low
    ins = (d1.high < d1.high.shift(1)) & (d1.low > d1.low.shift(1))
    ctx["id_nr4"] = (ins & (rng == rng.rolling(4).min())).shift(1)
    ctx["inside2"] = (ins & ins.shift(1)).shift(1)
    ctx["nr7_or_inside"] = (ins | (rng == rng.rolling(7).min())).shift(1)
    g1 = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                       max_trades_day=2, total_stop_pct=100.0, total_derisk_pct=100.0)
    rows, cand = [], {}
    for pat in ("inside", "nr7", "nr4", "id_nr4", "inside2", "nr7_or_inside"):
        for trend in (True, False):
            s = signals(m15, ctx, "nr", pattern=pat, trend=trend)
            if len(s) < 20:
                continue
            if not trend:
                s = s[s.dir == 1]
            s["leg"] = f"{pat}{'_T' if trend else '_L'}"
            t = engine.run(x, s, g1).trades
            h1, h2 = t[t.entry_time.dt.year <= 2022].R, t[t.entry_time.dt.year >= 2023].R
            tt = lambda v: round(v.mean() / v.std() * np.sqrt(len(v)), 2) if len(v) > 2 else 0  # noqa: E731
            rows.append({"leg": s.leg.iloc[0], "n": len(t), "avgR": round(t.R.mean(), 3), "t": tt(t.R),
                         "R_yr": round(t.R.sum() / 7.7, 1), "t_19_22": tt(h1), "t_23_26": tt(h2),
                         "yrs_pos": int((t.groupby(t.entry_time.dt.year).R.sum() > 0).sum())})
            cand[s.leg.iloc[0]] = s
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    base = book_signals(legs, A)
    out = [book_row(x, base, "best13")[0]]
    for name in ("nr4_T", "id_nr4_T", "inside2_T", "inside_L", "nr7_L"):
        if name in cand:
            s = cand[name].copy(); s["trail"] = 1.5 * s.sl
            out.append(book_row(x, pd.concat([base, s]).sort_index(kind="stable"), f"best13+{name}")[0])
            print(out[-1], flush=True)
    print(pd.DataFrame(out)[["book", "n", "R_yr", "SR", "DD%", "dDD%", "weekR", "pass250", "fail", "yrs_pos"]].to_string(index=False))
    (lab.REPORTS / "EXP-076").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-076" / "legs.csv", index=False)

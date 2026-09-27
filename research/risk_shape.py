"""EXP-074: raise the book's Sharpe by shaping risk (never above 0.5 % per trade).

  volT      : risk_mult = min(1, (median_vol / vol20)^p), vol20 = 20-day stdev of daily gold returns known at the
              signal time, median_vol = causal expanding median; p = 0.5 / 1
  legW      : walk-forward correlation-aware leg weights. Each year Y uses the per-leg daily P&L (R) of the
              book in the two previous years; weights = inverse-variance with Ledoit-Wolf-like shrinkage of the
              correlation matrix (min-variance tilt), scaled so the largest weight is 1; legs with negative
              trailing mean get 0.5 at most.
  asiaCap   : at most N of the Asia-open longs (drift, strong_close, friday, season, tom) per day.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, legcache, metrics  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.news_scope import book_row, funded_row  # noqa: E402
from research.payout_sim import A  # noqa: E402

ASIA = ["strong_close", "friday", "season", "tom", "drift"]   # priority order


def vol_mult(sig, d1, p):
    r = np.log(d1.close).diff()
    v = r.rolling(20).std().shift(1)                      # known before today's open
    med = v.expanding(250).median()
    m = np.minimum(1.0, (med / v) ** p).clip(lower=0.25)
    return m.reindex(sig.index.normalize()).fillna(1.0).to_numpy()


def leg_weights(trades, years, shrink=0.5):
    t = trades.assign(day=trades.exit_time.dt.normalize())
    P = t.pivot_table(index="day", columns="leg", values="R", aggfunc="sum").fillna(0.0)
    W = {}
    for y in years:
        w = P[(P.index.year >= y - 2) & (P.index.year <= y - 1)]
        sd = w.std().replace(0, np.nan)
        C = w.corr().fillna(0.0).to_numpy()
        C = (1 - shrink) * C + shrink * np.eye(len(C))
        cov = np.outer(sd.fillna(sd.mean()), sd.fillna(sd.mean())) * C
        mu = w.mean().to_numpy()
        raw = np.linalg.solve(cov, np.maximum(mu, 0) + 1e-9)          # tangency weights on positive means
        raw = np.maximum(raw, 0)
        ww = raw / raw.max() if raw.max() > 0 else np.ones(len(raw))
        ww = np.clip(ww, 0.25, 1.0)
        W[y] = dict(zip(P.columns, ww))
    return pd.DataFrame(W)


if __name__ == "__main__":
    m1, legs = legcache.load()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    base = book_signals(legs, A)
    books = {"best13": base}
    for p in (0.5, 1.0):
        s = base.copy(); s["risk_mult"] = s.get("risk_mult", pd.Series(1.0, index=s.index)).fillna(1.0) * vol_mult(s, d1, p)
        books[f"volT_p{p}"] = s
    # walk-forward leg weights from the base book's own trades (2019-20 -> 2021, ...); 2019-20 unweighted
    t0 = engine.run(x, base, g(True, 6)).trades
    W = leg_weights(t0, range(2021, 2027))
    print("leg weights by year:\n", W.round(2).to_string())
    s = base.copy(); rm = s.get("risk_mult", pd.Series(1.0, index=s.index)).fillna(1.0).to_numpy().copy()
    for i, (ts, leg) in enumerate(zip(s.index, s.leg)):
        if ts.year in W.columns and leg in W.index:
            rm[i] *= W.at[leg, ts.year]
    s["risk_mult"] = rm; books["legW"] = s
    for N in (2, 3):
        s = base.copy(); day = s.index.normalize()
        isA = s.leg.isin(ASIA).to_numpy()
        pr = s.leg.map({k: i for i, k in enumerate(ASIA)}).fillna(99).to_numpy()
        keep = np.ones(len(s), bool)
        dfA = pd.DataFrame({"day": day, "pr": pr, "i": np.arange(len(s))})[isA]
        for _, gdf in dfA.groupby("day"):
            drop = gdf.sort_values("pr").i.to_numpy()[N:]
            keep[drop] = False
        books[f"asiaCap{N}"] = s[keep]
    rows, keep_r = [], {}
    for name, s in books.items():
        row, res, m = book_row(x, s, name)
        row.update(funded_row(x, s, 1.25))
        rows.append(row); keep_r[name] = (m, res)
        print(row, flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    (lab.REPORTS / "EXP-074").mkdir(exist_ok=True)
    W.to_csv(lab.REPORTS / "EXP-074" / "leg_weights.csv")
    lab.save_experiment("EXP-074", {"variants": list(books)}, {k: v[0] for k, v in keep_r.items()},
                        {k: v[1] for k, v in keep_r.items()})

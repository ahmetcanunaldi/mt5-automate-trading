"""EXP-024: intraday but session-long horizon — one decision per day, flat by end of day.

Decision time T (server): 10:00 (London open) or 16:30 (NY open). Exit at 23:30 or SL = s * ATR_D(14).
Costs become ~1-2 % of R. Signals (all known at T):
  mom1   sign of previous day return            mom5  sign of previous 5-day return
  asia   sign of today's move from 01:00 to T    h1    H1 EMA50/200 trend at T
  usd    minus sign of EURUSD... (USD proxy) move from 01:00 to T (gold moves opposite to USD)
Plus a walk-forward LightGBM on ~20 daily features (yearly expanding blocks, predicts P(close_EOD > entry)).
Reported per year 2019-2026 with the engine (M1 execution, guards, news).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr, ema  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def build_daily(T_min):
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    eur = pd.read_parquet(lab.DATA / "EURUSD_M1_2018.parquet")["close"]
    usdx = pd.read_parquet(lab.DATA / "USDX_M1_2018.parquet")["close"]
    xag = pd.read_parquet(lab.DATA / "XAGUSD_M1_2018.parquet")["close"]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"])
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    atr_d = atr(d1, 14).shift(1)
    c = m1["close"]
    e50, e200 = ema(h1.close, 50), ema(h1.close, 200)
    rows = []
    for day in d1.loc[A:B].index:
        if day.dayofweek > 4:
            continue
        t = day + pd.Timedelta(minutes=T_min)
        t_open = day + pd.Timedelta(hours=1, minutes=5)
        t_end = day + pd.Timedelta(hours=23, minutes=30)
        p_t, p_o, p_e = c.asof(t), c.asof(t_open), c.asof(t_end)
        prev = d1.loc[:day - pd.Timedelta(days=1)]
        if len(prev) < 30 or not np.isfinite(p_t) or not np.isfinite(p_e):
            continue
        hk = h1.index.searchsorted(t) - 1            # last closed H1 bar before t
        r = {"day": day, "t": t, "entry": p_t, "exit_eod": p_e, "atr_d": atr_d.get(day, np.nan),
             "mom1": np.sign(prev.close.iloc[-1] - prev.close.iloc[-2]),
             "mom5": np.sign(prev.close.iloc[-1] - prev.close.iloc[-6]),
             "mom20": np.sign(prev.close.iloc[-1] - prev.close.iloc[-21]),
             "asia": np.sign(p_t - p_o),
             "asia_atr": (p_t - p_o) / atr_d.get(day, np.nan),
             "h1": np.sign(e50.iloc[hk] - e200.iloc[hk]) if hk >= 200 else 0.0,
             "usd": -np.sign(usdx.asof(t) - usdx.asof(t_open)),
             "eur": np.sign(eur.asof(t) - eur.asof(t_open)),
             "xag": np.sign(xag.asof(t) - xag.asof(t_open)),
             "ret1": (prev.close.iloc[-1] / prev.close.iloc[-2] - 1) * 100,
             "ret5": (prev.close.iloc[-1] / prev.close.iloc[-6] - 1) * 100,
             "ret20": (prev.close.iloc[-1] / prev.close.iloc[-21] - 1) * 100,
             "rng1": (prev.high.iloc[-1] - prev.low.iloc[-1]) / atr_d.get(day, np.nan),
             "pos_prev_rng": (p_t - prev.low.iloc[-1]) / max(prev.high.iloc[-1] - prev.low.iloc[-1], 1e-9),
             "dow": day.dayofweek}
        rows.append(r)
    D = pd.DataFrame(rows).set_index("day")
    D["fwd"] = (D.exit_eod - D.entry) / D.atr_d          # EOD move in daily ATRs
    return D


def to_signals(D, direction, sl_atr=1.0, tp_atr=3.0):
    d = direction[direction != 0]
    s = pd.DataFrame({"dir": d.astype(int).to_numpy(), "sl": (sl_atr * D.atr_d.reindex(d.index)).to_numpy(),
                      "tp": (tp_atr * D.atr_d.reindex(d.index)).to_numpy(), "hold_min": 24 * 60, "be": 0.0,
                      "trail": 0.0}, index=pd.DatetimeIndex(D.t.reindex(d.index).to_numpy()))
    return s.dropna()


def main():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(m1.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(m1, 1, blk, flt)
    g = engine.Guards(total_stop_pct=100.0, total_derisk_pct=100.0, flatten_min=23 * 60 + 30)
    out = []
    for T_min, tname in ((600, "10:00"), (990, "16:30")):
        D = build_daily(T_min)
        # the raw information content: mean EOD move (in ATR) signed by each signal
        print(f"\n### decision {tname}: {len(D)} days. mean signed EOD move (ATR units), t-stat, by signal")
        for sgn in ("mom1", "mom5", "mom20", "asia", "h1", "usd", "eur", "xag"):
            v = (D[sgn] * D.fwd).replace([np.inf, -np.inf], np.nan).dropna()
            v = v[D[sgn].reindex(v.index) != 0]
            print(f"   {sgn:<6} mean {v.mean():+.3f}  t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}  "
                  f"by year {v.groupby(v.index.year).mean().round(2).to_dict()}")
        # walk-forward LightGBM on daily features
        import lightgbm as lgb
        feats = ["mom1", "mom5", "mom20", "asia", "asia_atr", "h1", "usd", "eur", "xag", "ret1", "ret5", "ret20",
                 "rng1", "pos_prev_rng", "dow"]
        y = (D.fwd > 0).astype(int)
        pred = pd.Series(np.nan, index=D.index)
        for yr in range(2021, 2027):
            tr = D.index.year < yr; te = D.index.year == yr
            mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            mdl.fit(D.loc[tr, feats], y[tr])
            pred[te] = mdl.predict_proba(D.loc[te, feats])[:, 1]
        m = pred.notna()
        v = (np.sign(pred[m] - 0.5) * D.fwd[m])
        print(f"   ML(2021+) mean {v.mean():+.3f} t {v.mean() / v.std() * np.sqrt(len(v)):+.2f} "
              f"by year {v.groupby(v.index.year).mean().round(2).to_dict()}")
        # engine runs for the simple signals and the ML book
        books = {s: D[s] for s in ("mom1", "asia", "h1", "usd")}
        books["usd&asia"] = D.asia.where(D.asia == D.usd, 0)
        books["ML"] = np.sign(pred - 0.5).fillna(0)
        for bn, dirn in books.items():
            sig = to_signals(D, dirn)
            res = engine.run(x, sig, g)
            mm = metrics.summarize(res, f"{tname}|{bn}")
            t = res.trades
            yr = t.groupby(t.entry_time.dt.year)["R"].sum().round(1).to_dict()
            out.append({"T": tname, "book": bn, "n": mm["trades"], "avgR": mm["avg_R"], "SR": mm["sharpe"],
                        "PF": mm["profit_factor"], "net%": mm["net_pct"], "DD%": mm["max_total_dd_pct"],
                        "yrs_pos": sum(v > 0 for v in yr.values()), "R_by_year": yr})
            print("   ", out[-1], flush=True)
    T = pd.DataFrame(out)
    (lab.REPORTS / "EXP-024").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-024" / "summary.csv", index=False)


if __name__ == "__main__":
    main()

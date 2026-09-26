"""EXP-021: what separates the trend-breakout regime (2022-25) from 2019-21?

Take the rule portfolio A (EXP-013 params, no total stop, K1) over 2019-2026 on M1, aggregate R per week/month,
and relate it to regime features computed ONLY from data before that week/month:
  rv20      20-day realized vol of daily returns (annualized %)
  rng_pct   mean daily high-low range / close over the previous 20 days (%)
  trend60   |60-day log return| / (daily vol * sqrt(60))       (trend t-stat)
  er20/er60 efficiency ratio of daily closes
  ac_m15    lag-1 autocorrelation of M15 returns over the previous 20 days (intraday persistence)
  vr16      variance ratio of M15 returns at 16 bars (4 h) over the previous 20 days (>1 = trending intraday)
  perf_4w / perf_12w  the strategy's own trailing R (equity-curve filter)
Tercile thresholds are EXPANDING (only past weeks), so the split is causal.
"""
import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, features, lab  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.oos import CANDS  # noqa: E402
from research.portfolio import leg_signals  # noqa: E402

A, B = "2019-01-01", "2026-09-26"


def load_all():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"])
    return m1, m15, h1, d1


def exec_all(m1):
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    return engine.prepare_exec(ex, 1, blk, flt)


def regime_features(m15, d1):
    r = np.log(d1["close"]).diff()
    f = pd.DataFrame(index=d1.index)
    f["rv20"] = r.rolling(20).std() * np.sqrt(252) * 100
    f["rng_pct"] = ((d1["high"] - d1["low"]) / d1["close"] * 100).rolling(20).mean()
    f["trend60"] = (np.log(d1["close"]).diff(60)).abs() / (r.rolling(60).std() * np.sqrt(60))
    for n in (20, 60):
        f[f"er{n}"] = (d1["close"].diff(n).abs() / d1["close"].diff().abs().rolling(n).sum())
    q = np.log(m15["close"]).diff()
    day = q.index.normalize()
    # daily stats of intraday M15 returns, then 20-day rolling
    g = pd.DataFrame({"r": q, "r1": q.shift(1)}).dropna()
    daily = g.groupby(g.index.normalize()).apply(lambda x: pd.Series({"sxy": (x.r * x.r1).sum(), "sxx": (x.r ** 2).sum(),
                                                                         "n": len(x)}))
    f["ac_m15"] = (daily["sxy"].rolling(20).sum() / daily["sxx"].rolling(20).sum()).reindex(f.index)
    s16 = q.rolling(16).sum()
    v16 = (s16 ** 2).groupby(day).mean().rolling(20).mean()
    v1 = (q ** 2).groupby(day).mean().rolling(20).mean()
    f["vr16"] = (v16 / (16 * v1)).reindex(f.index)
    return f.shift(1)          # known at the START of each day


def main():
    m1, m15, h1, d1 = load_all()
    df = features.base_frame(m15, h1)
    x = exec_all(m1)
    legs, g = CANDS["A_base3_K1"]
    g = replace(g, total_stop_pct=100.0, total_derisk_pct=100.0)
    res = engine.run(x, leg_signals(df, legs).loc[A:B], g)
    t = res.trades
    F = regime_features(m15, d1)
    out = {}
    for freq, lab_ in (("W-FRI", "week"), ("MS", "month")):
        R = t.set_index("entry_time")["R"].resample(freq).sum()
        n = t.set_index("entry_time")["R"].resample(freq).count()
        Fp = F.resample(freq).first().reindex(R.index)            # regime at the start of the period
        Fp["perf_4"] = R.shift(1).rolling(4).sum()
        Fp["perf_12"] = R.shift(1).rolling(12).sum()
        D = Fp.assign(R=R, n=n).dropna()
        print(f"\n=== per {lab_}: {len(D)} periods, mean R {D.R.mean():.2f} ===")
        rows = []
        for col in [c for c in D.columns if c not in ("R", "n")]:
            s = D[col]
            # expanding (causal) tercile thresholds, need >= 26 past periods
            lo = s.expanding(26).quantile(1 / 3).shift(1)
            hi = s.expanding(26).quantile(2 / 3).shift(1)
            ok = lo.notna()
            top = D.R[ok & (s > hi)]; bot = D.R[ok & (s < lo)]; mid = D.R[ok & (s >= lo) & (s <= hi)]
            rows.append({"feature": col, "spearman": round(s.corr(D.R, method="spearman"), 3),
                         "R_low": round(bot.mean(), 2), "R_mid": round(mid.mean(), 2), "R_high": round(top.mean(), 2),
                         "t_high_vs_low": round((top.mean() - bot.mean()) /
                                                np.sqrt(top.var() / len(top) + bot.var() / len(bot)), 2)})
        T = pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False)
        print(T.to_string(index=False))
        out[lab_] = (D, T)
    yr = t.groupby(t.entry_time.dt.year)["R"].sum().round(1).to_dict()
    Fy = F.groupby(F.index.year).mean().round(3)
    Fy["R"] = pd.Series(yr)
    print("\n=== yearly regime profile ===\n", Fy.loc[2019:].to_string())
    (lab.REPORTS / "EXP-021").mkdir(exist_ok=True)
    out["week"][0].to_csv(lab.REPORTS / "EXP-021" / "weekly.csv")
    return out, t, F


if __name__ == "__main__":
    main()

"""EXP-042: post-news continuation / fade. At T+30 min after a US high-impact release (entries are allowed again),
if |move(T-1min -> T+30)| >= k * ATR_D, trade WITH (cont) or AGAINST (fade) it; hold h minutes (max end of day),
SL = sl * ATR_D. Event groups: all, tier-1 (NFP, CPI, FOMC, PCE, GDP, retail sales, ISM). M1 2019-2026, $100k."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
TIER1 = ["nonfarm-payrolls", "consumer-price-index-mm", "consumer-price-index-ex-food-energy-mm", "fed-interest-rate-decision",
         "core-pce-price-index-mm", "gross-domestic-product-qq", "retail-sales-mm", "ism-manufacturing-pmi",
         "ism-non-manufacturing-pmi", "fomc-press-conference", "producer-price-index-mm", "average-hourly-earnings-mm"]

if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    c = m1["close"]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    atr_d = atr(d1, 14).shift(1)
    cal = pd.read_csv(lab.DATA / "calendar_usd_high.csv")
    cal["t"] = calendar_news.utc_to_server(pd.DatetimeIndex(pd.to_datetime(cal.time_server) - pd.Timedelta(hours=3)))
    cal = cal[(cal.t >= A) & (cal.t <= B)]
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    # entries at T+30 exactly are allowed; later news inside the hold window still forces a flatten
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 29, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    g = engine.Guards(initial_balance=100_000, max_trades_day=4, total_stop_pct=100.0, total_derisk_pct=100.0)
    rows, keep = [], {}
    for grp, mode, k, sl, hold in itertools.product(["all", "tier1"], ["cont", "fade"], [0.1, 0.2, 0.3], [0.5, 1.0],
                                                    [60, 240, 600]):
        ev = cal if grp == "all" else cal[cal.event_code.isin(TIER1)]
        T = pd.DatetimeIndex(sorted(set(ev.t)))
        p0 = c.asof(T - pd.Timedelta(minutes=1)).to_numpy(); p1 = c.asof(T + pd.Timedelta(minutes=30)).to_numpy()
        ad = atr_d.reindex(T.normalize()).to_numpy()
        mv = p1 - p0
        big = np.abs(mv) >= k * ad
        d = np.sign(mv) * (1 if mode == "cont" else -1)
        s = pd.DataFrame({"dir": d[big].astype(int), "sl": sl * ad[big], "tp": 1e6, "hold_min": hold, "be": 0.0,
                          "trail": 0.0, "leg": "postnews"}, index=T[big] + pd.Timedelta(minutes=30)).dropna()
        s = s[~s.index.duplicated()]
        if len(s) < 30:
            continue
        res = engine.run(x, s, g)
        t = res.trades
        if len(t) < 20:
            continue
        m = metrics.summarize(res, "")
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"{grp}|{mode}|k{k}|sl{sl}|h{hold}"
        rows.append({"cfg": name, "n": len(t), "wr": m["win_rate"], "avgR": m["avg_R"],
                     "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "R_yr": round(t.R.sum() / 7.7, 1),
                     "SR": m["sharpe"], "yrs_pos": int((yr > 0).sum()), **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
    R = pd.DataFrame(rows).sort_values("t", ascending=False)
    (lab.REPORTS / "EXP-042").mkdir(exist_ok=True)
    R.to_csv(lab.REPORTS / "EXP-042" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.head(20).to_string(index=False))
    print(R.groupby(R.cfg.str.split("|").str[1]).t.describe()[["count", "mean", "max", "min"]].round(2))

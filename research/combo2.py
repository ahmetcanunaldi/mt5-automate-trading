"""EXP-038: bigger diversified portfolio of every positive-family leg (2019-2026, M1 exec, $100k, real guards):
EXP-035 legs + Larry-Williams breakout (k0.4 from 10:00) + NR7/inside-day breakout with trend + Friday-close drift."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import A, B, build_legs  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_volbreak import daily_ctx, signals as vb_signals  # noqa: E402

if __name__ == "__main__":
    m1, legs = build_legs()
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    ctx = daily_ctx(m1)
    lw = vb_signals(m15.loc[A:B], ctx, "lw", k=0.4, sl=1.0, start_min=600); lw["leg"] = "lw"
    nr = vb_signals(m15.loc[A:B], ctx, "nr", pattern="inside", trend=True); nr["leg"] = "inside"
    nr7 = vb_signals(m15.loc[A:B], ctx, "nr", pattern="nr7", trend=True); nr7["leg"] = "nr7"
    a15 = atr(m15, 14)
    days = pd.DatetimeIndex(np.unique(m1.loc[A:B].index.normalize()))
    fri = days[days.dayofweek == 4] + pd.Timedelta(minutes=1385)
    fc = pd.DataFrame({"dir": 1, "sl": 4.0 * a15.asof(fri - pd.Timedelta(minutes=15)).to_numpy(), "tp": 1e6,
                       "hold_min": 50, "be": 0.0, "trail": 0.0, "leg": "fri_close"}, index=fri).dropna()
    legs.update({"lw": lw, "inside": nr, "nr7": nr7, "fri_close": fc})
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    sets = {
        "prev5": ["trendH4", "tday", "drift", "friday", "tom"],
        "prev5+lw": ["trendH4", "tday", "drift", "friday", "tom", "lw"],
        "prev5+nr": ["trendH4", "tday", "drift", "friday", "tom", "inside", "nr7"],
        "all": ["trendH4", "tday", "drift", "friday", "tom", "lw", "inside", "nr7", "fri_close"],
        "intraday_all": ["tday", "drift", "friday", "lw", "inside", "nr7", "fri_close"],
    }
    rows, keep = [], {}
    for (sn, names), K, wf in itertools.product(sets.items(), [4, 6], [False, True]):
        s = pd.concat([legs[n] for n in names]).sort_index(kind="stable")
        intraday_only = sn == "intraday_all"
        g = engine.Guards(initial_balance=100_000, intraday=intraday_only, weekend_flat=wf, first_entry_min=65,
                          last_entry_min=1390, flatten_min=1437 if intraday_only else 1425,
                          fri_flatten_min=1500 if intraday_only else 1350, max_trades_day=8, max_positions=K,
                          max_open_risk_pct=0.5 * K)
        res = engine.run(x, s, g)
        t = res.trades
        m = metrics.summarize(res, sn); m.update(metrics.monte_carlo_dd(res))
        ch = metrics.challenge_sim(res, max_days=250)
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        name = f"{sn}|K{K}|wf{int(wf)}"
        rows.append({"cfg": name, "n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / 7.7, 1), "SR": m["sharpe"],
                     "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
                     "mc95": m["mc_dd95_pct"], "pass250": ch["pass_rate"], "med_days": ch["median_days_to_pass"],
                     "yrs_pos": int((yr > 0).sum()), **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("SR", ascending=False)
    (lab.REPORTS / "EXP-038").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-038" / "summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(T.to_string(index=False))
    top = list(T.cfg.head(2))
    lab.save_experiment("EXP-038", {"sets": sets}, {c: keep[c][0] for c in top},
                        {c.replace("|", "_").replace("+", "-"): keep[c][1] for c in top})

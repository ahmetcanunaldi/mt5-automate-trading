"""EXP-044: cost-aware minimum stop. The round-trip cost is a fixed ~$0.37/oz, so a leg whose stop is small pays a
large share of R in low-vol years. Rule (not fitted to years): skip a trade when its stop < min_sl dollars,
i.e. cost/stop > 0.37/min_sl. Applied to the MTF leg (EXP-043) and to every leg of the EXP-038 portfolio."""
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
    legs["lw"] = vb_signals(m15.loc[A:B], ctx, "lw", k=0.4, sl=1.0, start_min=600)
    legs["inside"] = vb_signals(m15.loc[A:B], ctx, "nr", pattern="inside", trend=True)
    legs["nr7"] = vb_signals(m15.loc[A:B], ctx, "nr", pattern="nr7", trend=True)
    for k in legs:
        legs[k]["leg"] = k
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    print("median stop $ by leg:", {k: round(v.sl.median(), 1) for k, v in legs.items()})
    names = ["trendH4", "tday", "drift", "friday", "tom", "lw", "inside", "nr7"]
    rows = []
    for min_sl in (0.0, 3.0, 5.0, 7.5, 10.0):
        s = pd.concat([legs[n][legs[n].sl >= min_sl] for n in names]).sort_index(kind="stable")
        g = engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=False, last_entry_min=1390,
                          max_trades_day=8, max_positions=6, max_open_risk_pct=3.0)
        res = engine.run(x, s, g)
        t = res.trades
        m = metrics.summarize(res, ""); m.update(metrics.monte_carlo_dd(res))
        ch = metrics.challenge_sim(res, max_days=250)
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        rows.append({"min_sl$": min_sl, "n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / 7.7, 1),
                     "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "mc95": m["mc_dd95_pct"],
                     "pass250": ch["pass_rate"], "med": ch["median_days_to_pass"], "yrs_pos": int((yr > 0).sum()),
                     **{f"R{y}": round(v, 1) for y, v in yr.items()}})
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    (lab.REPORTS / "EXP-044").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-044" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))

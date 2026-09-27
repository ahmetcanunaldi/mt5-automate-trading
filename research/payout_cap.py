"""EXP-071: daily profit cap to satisfy the 35 % consistency rule faster (payout at +3 %)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.combo8 import build  # noqa: E402
from research.payout_sim import A, chart, guards, summarize_payouts  # noqa: E402

if __name__ == "__main__":
    m1, legs = build()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    sig = book_signals(legs, A)
    rows, keep = [], {}
    for cap in (0.0, 0.75, 1.0, 1.25):
        res = engine.run(x, sig, guards(3.0, 35.0, cap))
        r, p = summarize_payouts(res, f"cap {cap} %")
        # performance of the same book without payouts (for R/yr, Sharpe)
        res_np = engine.run(x, sig, guards(0.0, 100.0, cap))
        m = metrics.summarize(res_np, "")
        r.update({"R_yr": round(res_np.trades.R.sum() / 7.7, 1), "SR": m["sharpe"], "worst_day%": m["max_daily_dd_pct"],
                  "peak_DD%": m["max_total_dd_pct"], "week_R": m["week_R_mean"]})
        rows.append(r); keep[cap] = (res, p)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    best = max(keep, key=lambda k: len(keep[k][1]))
    res, p = keep[best]
    d = lab.REPORTS / "EXP-071"; d.mkdir(exist_ok=True)
    T.to_csv(d / "summary.csv", index=False); p.to_csv(d / f"payouts_cap{best}.csv", index=False)
    print(f"\nbest cap {best}: payouts per year:", p.groupby(p.time.dt.year).amount.agg(["count", "sum"]).round(0).to_dict())
    chart(res, p, d / f"payouts_cap{best}.png",
          f"EXP-071 best13 funded 2019–26, day cap {best} %: {len(p)} payouts of ≥ +3 %, median every {p.days.median():.0f} days")

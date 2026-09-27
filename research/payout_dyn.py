"""EXP-077: consistency-aware daily profit cap for the funded phase. The 35 % rule needs best day <= 35 % of the
cycle profit. A cap of cons/(100-cons) x the cycle profit made before today (= 0.538 x P0) keeps every day inside
the rule, with a floor (fixed cap) so that early days can still reach the payout threshold."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, legcache, metrics  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.payout_sim import A, chart, guards, summarize_payouts  # noqa: E402

if __name__ == "__main__":
    m1, legs = legcache.load()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    sig = book_signals(legs, A)
    rows, keep = [], {}
    for floor, dyn in ((1.25, False), (0.75, True), (1.05, True), (1.25, True), (1.5, True)):
        gp = dataclasses.replace(guards(3.0, 35.0, floor), cap_dynamic=dyn)
        res = engine.run(x, sig, gp)
        r, p = summarize_payouts(res, f"{'dyn' if dyn else 'fixed'} cap floor {floor} %")
        m = metrics.summarize(engine.run(x, sig, dataclasses.replace(gp, payout_pct=0.0, consistency_pct=35.0)), "")
        r.update({"SR": m["sharpe"], "worst_day%": m["max_daily_dd_pct"], "week_R": m["week_R_mean"]})
        rows.append(r); keep[r["case"]] = (res, p)
        print(r, flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    d = lab.REPORTS / "EXP-077"; d.mkdir(exist_ok=True)
    T.to_csv(d / "summary.csv", index=False)
    best = max(keep, key=lambda k: (len(keep[k][1]), keep[k][1].amount.sum() if len(keep[k][1]) else 0))
    res, p = keep[best]
    p.to_csv(d / "payouts_best.csv", index=False)
    print("best:", best, p.groupby(p.time.dt.year).amount.agg(["count", "sum"]).round(0).to_dict())
    chart(res, p, d / "payouts_best.png", f"EXP-077 best13 funded 2019–26, {best}: {len(p)} payouts, "
          f"median every {p.days.median():.0f} days, ${p.amount.sum():,.0f}")

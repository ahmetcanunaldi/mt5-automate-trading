"""EXP-072: re-open positions closed by the news flatten once the blackout (+30 min) is over, keeping the original
stop distance, target and expiry. Half of best13's exits are NEWS flattens (2,063 / 4,058), so the legs rarely
hold their full intended window. Compared in challenge mode (compounding) and funded mode (payouts, 35 %
consistency, day caps)."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.combo8 import build  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.payout_sim import A, guards as pguards, summarize_payouts  # noqa: E402


def evaluate(x, sig, reent, label, caps=(0.0, 1.25)):
    gc = dataclasses.replace(g(True, 6), news_reentry=reent)
    res = engine.run(x, sig, gc)
    m = metrics.summarize(res, label); m.update(metrics.monte_carlo_dd(res))
    ch = metrics.challenge_sim(res, max_days=250)
    t = res.trades
    row = {"book": label, "n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / 7.7, 1), "SR": m["sharpe"],
           "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"],
           "weekR": m["week_R_mean"], "pass250": ch["pass_rate"], "fail": ch["fail_rate"],
           "med": ch["median_days_to_pass"], "yrs_pos": int((t.groupby(t.entry_time.dt.year).R.sum() > 0).sum())}
    for cap in caps:
        gp = dataclasses.replace(pguards(3.0, 35.0, cap), news_reentry=reent)
        r, _ = summarize_payouts(engine.run(x, sig, gp), "")
        gn = dataclasses.replace(pguards(0.0, 100.0, cap), news_reentry=reent)
        mf = metrics.summarize(engine.run(x, sig, gn), "")
        row.update({f"c{cap}_pay": r["payouts"], f"c{cap}_med_d": r.get("median_days"),
                    f"c{cap}_$": r.get("withdrawn_$"), f"c{cap}_SR": mf["sharpe"], f"c{cap}_wkR": mf["week_R_mean"]})
    return row, res, m


if __name__ == "__main__":
    m1, legs = build()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    sig = book_signals(legs, A)
    rows, keep = [], {}
    for reent in (False, True):
        label = f"best13{' +news_reentry' if reent else ''}"
        row, res, m = evaluate(x, sig, reent, label)
        rows.append(row); keep[label] = (m, res)
        t = res.trades
        print(label, "\n", t.groupby("leg").R.agg(["count", "sum", "mean"]).round(3).T.to_string(), flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    lab.save_experiment("EXP-072", {"change": "news re-entry"}, {k: v[0] for k, v in keep.items()},
                        {k.replace(" ", "_").replace("+", ""): v[1] for k, v in keep.items()})

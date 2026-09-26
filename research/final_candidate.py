"""EXP-047: candidate portfolio (9 positive-family legs, equal 0.5 % risk) across windows and weekend rules,
plus leg-drop sensitivity (remove one leg at a time) on 2019-2026."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402

LEGS9 = ["trendH4", "tday", "drift", "friday", "tom", "lw", "inside", "nr7", "fri_close"]


def g(wf, K=6):
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=wf, last_entry_min=1390,
                         flatten_min=1425, fri_flatten_min=1350, max_trades_day=8, max_positions=K,
                         max_open_risk_pct=0.5 * K)


def book(x, legs, names, wf, K=6):
    s = pd.concat([legs[n] for n in names]).sort_index(kind="stable")
    res = engine.run(x, s, g(wf, K))
    m = metrics.summarize(res, ""); m.update(metrics.monte_carlo_dd(res))
    ch = metrics.challenge_sim(res, max_days=250)
    t = res.trades
    yrs = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    return {"n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / yrs, 1), "SR": m["sharpe"],
            "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"],
            "pass250": ch["pass_rate"], "fail": ch["fail_rate"], "med": ch["median_days_to_pass"],
            "yrs_pos": int((t.groupby(t.entry_time.dt.year).R.sum() > 0).sum())}, res, m


if __name__ == "__main__":
    m1, legs = all_legs()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    X = {"2019-26": engine.prepare_exec(ex, 1, blk, flt)}
    k21 = ex.index >= "2021-01-01"
    X["2021-26"] = engine.prepare_exec(ex.loc["2021":], 1, blk[k21], flt[k21])
    rows, keep = [], {}
    for (wn, x), wf, K in itertools.product(X.items(), [False, True], [4, 6]):
        r, res, m = book(x, {k: v.loc[wn[:4]:] for k, v in legs.items()}, LEGS9, wf, K)
        name = f"{wn}|wf{int(wf)}|K{K}"
        rows.append({"cfg": name, **r}); keep[name] = (m, res)
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False))
    drop = []
    for leg in LEGS9:
        r, _, _ = book(X["2019-26"], legs, [n for n in LEGS9 if n != leg], True, 6)
        drop.append({"dropped": leg, **r})
    D = pd.DataFrame(drop)
    print("\nleave-one-leg-out (2019-26, weekend flat, K6):\n", D.to_string(index=False))
    (lab.REPORTS / "EXP-047").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(lab.REPORTS / "EXP-047" / "windows.csv", index=False)
    D.to_csv(lab.REPORTS / "EXP-047" / "leave_one_out.csv", index=False)
    sel = ["2019-26|wf1|K6", "2021-26|wf1|K6"]
    lab.save_experiment("EXP-047", {"legs": LEGS9}, {k: keep[k][0] for k in sel}, {k: keep[k][1] for k in sel})

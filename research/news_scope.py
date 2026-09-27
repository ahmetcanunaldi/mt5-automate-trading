"""EXP-073: how much does the news rule cost, and which events really matter?

Our calendar (MT5 "high importance", 35 USD event codes) is much broader than the red-folder list prop firms use:
it includes EIA crude stocks, jobless claims, S&P/Markit PMIs, bond auctions, every Powell speech. Half of best13's
exits are news flattens. Variants (entries always blocked -30/+30 min around the events in scope):
  all_flat10   : current rule — all 35 codes, flatten 10 min before (baseline)
  tier1_flat10 : tier-1 macro releases only, flatten 10 min before
  all_hold     : all codes, entries blocked but open positions are held through the release
  tier1_hold   : tier-1 only, hold through
"""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, legcache, metrics  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.payout_sim import A, guards as pguards, summarize_payouts  # noqa: E402

TIER1 = {"nonfarm-payrolls", "unemployment-rate", "average-hourly-earnings-mm", "consumer-price-index",
         "consumer-price-index-yy", "consumer-price-index-mm", "consumer-price-index-ex-food-energy-mm",
         "consumer-price-index-ex-food-energy-nsa-mm", "fed-interest-rate-decision", "fomc-press-conference",
         "fed-chair-powell-testimony", "retail-sales-mm", "retail-sales-ex-autos-mm", "ism-manufacturing-pmi",
         "ism-non-manufacturing-pmi", "gross-domestic-product-qq", "producer-price-index-mm", "jolts-job-openings",
         "adp-nonfarm-employment-change"}


def news_times(codes=None):
    cal = pd.read_csv(calendar_news.CAL_PATH)
    if codes is not None:
        cal = cal[cal.event_code.isin(codes)]
    t_utc = pd.DatetimeIndex(pd.to_datetime(cal["time_server"]) - pd.Timedelta(hours=3))
    return pd.DatetimeIndex(np.unique(calendar_news.utc_to_server(t_utc)))


def book_row(x, sig, label, gc=None):
    res = engine.run(x, sig, gc or g(True, 6))
    m = metrics.summarize(res, label); m.update(metrics.monte_carlo_dd(res))
    ch = metrics.challenge_sim(res, max_days=250)
    t = res.trades
    row = {"book": label, "n": len(t), "avgR": m["avg_R"], "R_yr": round(t.R.sum() / 7.7, 1), "SR": m["sharpe"],
           "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"],
           "weekR": m["week_R_mean"], "wk>=2R": m["weeks_ge_2R"], "pass250": ch["pass_rate"], "fail": ch["fail_rate"],
           "med": ch["median_days_to_pass"], "yrs_pos": int((t.groupby(t.entry_time.dt.year).R.sum() > 0).sum()),
           "news_exits": int((t.reason == "NEWS").sum())}
    return row, res, m


def funded_row(x, sig, cap):
    r, _ = summarize_payouts(engine.run(x, sig, pguards(3.0, 35.0, cap)), "")
    mf = metrics.summarize(engine.run(x, sig, pguards(0.0, 100.0, cap)), "")
    return {f"cap{cap}_pay": r["payouts"], f"cap{cap}_med_d": r.get("median_days"), f"cap{cap}_mean_d": r.get("mean_days"),
            f"cap{cap}_$": r.get("withdrawn_$"), f"cap{cap}_SR": mf["sharpe"], f"cap{cap}_wkR": mf["week_R_mean"]}


if __name__ == "__main__":
    m1, legs = legcache.load()
    ex = m1.loc[A:B]
    sig = book_signals(legs, A)
    scopes = {"all": news_times(), "tier1": news_times(TIER1)}
    print({k: len(v) for k, v in scopes.items()})
    rows, keep = [], {}
    for scope, flat in (("all", True), ("tier1", True), ("all", False), ("tier1", False)):
        blk, flt = calendar_news.blackout_masks(ex.index, 1, scopes[scope], 30, 30, 10)
        if not flat:
            flt = np.zeros_like(flt)
        x = engine.prepare_exec(ex, 1, blk, flt)
        label = f"{scope}_{'flat10' if flat else 'hold'}"
        row, res, m = book_row(x, sig, label)
        for cap in (0.0, 1.25):
            row.update(funded_row(x, sig, cap))
        rows.append(row); keep[label] = (m, res)
        print(row, flush=True)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    lab.save_experiment("EXP-073", {"tier1": sorted(TIER1)}, {k: v[0] for k, v in keep.items()},
                        {k: v[1] for k, v in keep.items()})

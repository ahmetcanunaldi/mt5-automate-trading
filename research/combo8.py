"""EXP-065: best12 = best10 + strong_close + drift(low-vol only), plus the season leg at 0.5x / 1.0x risk."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.features import atr  # noqa: E402
from research.final_candidate import LEGS9, g  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_trend_intraday import scan, setup as td_setup  # noqa: E402
from research.strong_close import signals as sc_signals  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402


def build():
    m1, legs = all_legs()
    m15_td, day, _ = td_setup()
    s = scan(m15_td, day, "tday", 0.3, 1.0, None, 900); s["hold_min"] = 510; s["leg"] = "tday900"
    legs["tday900"] = s.loc[A:B]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    legs["strong_close"] = sc_signals(d1, 0.6, 1.0).loc[A:B]
    atr_bps = (atr(d1, 14) / d1.close * 1e4).shift(1)
    low = atr_bps <= atr_bps.expanding(120).median().shift(1)
    dr = legs["drift"]; legs["drift"] = dr[low.reindex(dr.index.normalize()).fillna(False).to_numpy()]
    ad = atr(d1, 14).shift(1)
    days = d1.loc[A:B].index; md = days[days.month.isin([1, 7, 8])]
    legs["season"] = pd.DataFrame({"dir": 1, "sl": ad.reindex(md).to_numpy(), "tp": 1e6, "hold_min": 1360, "be": 0.0,
                                   "trail": 0.0, "leg": "season"}, index=md + pd.Timedelta(minutes=67)).dropna()
    return m1, legs


BEST12 = LEGS9 + ["tday900", "strong_close"]

if __name__ == "__main__":
    m1, legs = build()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows, keep = [], {}
    for name, rm in (("best12", None), ("best12+season0.5", 0.5), ("best12+season1.0", 1.0)):
        parts = [legs[n] for n in BEST12]
        if rm:
            sz = legs["season"].copy(); sz["risk_mult"] = rm; parts.append(sz)
        sig = pd.concat(parts).sort_index(kind="stable")
        res = engine.run(x, sig, g(True, 6))
        m = metrics.summarize(res, name); m.update(metrics.monte_carlo_dd(res))
        m["challenge"] = metrics.challenge_sim(res, max_days=250); m["gates"] = metrics.gate_check(m)
        eq = res.daily["end"]
        rows.append({"book": name, "n": m["trades"], "R_yr": round(res.trades.R.sum() / 7.7, 1), "SR": m["sharpe"],
                     "CAGR%": m["cagr_pct"], "net$": round(eq.iloc[-1] - 100_000), "DD%": m["max_total_dd_pct"],
                     "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"], "week_R": m["week_R_mean"],
                     "wk>=2R": m["weeks_ge_2R"], "pass250": m["challenge"]["pass_rate"],
                     "fail": m["challenge"]["fail_rate"], "med": m["challenge"]["median_days_to_pass"]})
        keep[name] = (m, res)
    print(pd.DataFrame(rows).to_string(index=False))
    lab.save_experiment("EXP-065", {"legs": BEST12 + ["season"]}, {k: v[0] for k, v in keep.items()},
                        {k: v[1] for k, v in keep.items()})

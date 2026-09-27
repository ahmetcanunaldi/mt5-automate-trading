"""EXP-062: new legs on top of best10+strong_close: month-season long (Jan/Jul/Aug, open->close daily) and a
3-day strong-close swing variant; M1 2019-2026, weekend flat, K6, real guards."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.features import atr  # noqa: E402
from research.final_candidate import LEGS9, book  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.run_trend_intraday import scan, setup as td_setup  # noqa: E402
from research.strong_close import signals as sc_signals  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402

if __name__ == "__main__":
    m1, legs = all_legs()
    m15_td, day, _ = td_setup()
    s = scan(m15_td, day, "tday", 0.3, 1.0, None, 900); s["hold_min"] = 510; s["leg"] = "tday900"
    legs["tday900"] = s.loc[A:B]
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    legs["strong_close"] = sc_signals(d1, 0.6, 1.0).loc[A:B]
    sc3 = sc_signals(d1, 0.6, 1.5, hold_days=3).loc[A:B]; sc3["leg"] = "strong_close3"
    legs["strong_close3"] = sc3
    ad = atr(d1, 14).shift(1)
    days = d1.loc[A:B].index
    mdays = days[days.month.isin([1, 7, 8])]
    legs["season"] = pd.DataFrame({"dir": 1, "sl": 1.0 * ad.reindex(mdays).to_numpy(), "tp": 1e6, "hold_min": 1360,
                                   "be": 0.0, "trail": 0.0, "leg": "season"}, index=mdays + pd.Timedelta(minutes=67)).dropna()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    base = LEGS9 + ["tday900", "strong_close"]
    rows = []
    for name, names in (("best11", base), ("best11+season", base + ["season"]),
                        ("best10+strong_close3", LEGS9 + ["tday900", "strong_close3"]),
                        ("best11+season+sc3", base + ["season", "strong_close3"])):
        r, res, m = book(x, legs, names, True, 6)
        t = res.trades
        extra = {k: round(v, 1) for k, v in t.groupby("leg").R.sum().items() if k in ("season", "strong_close3", "strong_close")}
        rows.append({"book": name, **r, "week_R": m["week_R_mean"], "wk>=2R": m["weeks_ge_2R"], **extra})
    print(pd.DataFrame(rows).to_string(index=False))

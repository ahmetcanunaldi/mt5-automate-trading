"""EXP-064: strong-close leg exiting at 10:00 (Asia only) vs end of day; drift leg restricted to low-vol days;
portfolio impact (best11 = best10 + strong_close)."""
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
    for sl in (0.5, 1.0):
        s10 = sc_signals(d1, 0.6, sl).loc[A:B].copy(); s10["hold_min"] = 535; s10["leg"] = "strong_close"
        legs[f"sc10_sl{sl}"] = s10
    atr_bps = (atr(d1, 14) / d1.close * 1e4).shift(1)
    low = atr_bps <= atr_bps.expanding(120).median().shift(1)          # causal median
    dr = legs["drift"].copy()
    dr = dr[low.reindex(dr.index.normalize()).fillna(False).to_numpy()]
    legs["drift_lowvol"] = dr
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    base = LEGS9 + ["tday900"]
    rows = []
    for name, names in (("best11 (sc to 23:30)", base + ["strong_close"]),
                        ("best10 + sc to 10:00 sl0.5", base + ["sc10_sl0.5"]),
                        ("best10 + sc to 10:00 sl1.0", base + ["sc10_sl1.0"]),
                        ("best11 drift->lowvol", [n if n != "drift" else "drift_lowvol" for n in base] + ["strong_close"])):
        r, res, m = book(x, legs, names, True, 6)
        t = res.trades
        rows.append({"book": name, **r, "week_R": m["week_R_mean"], "wk>=2R": m["weeks_ge_2R"],
                     "sc_R": round(t[t.leg == "strong_close"].R.sum(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))

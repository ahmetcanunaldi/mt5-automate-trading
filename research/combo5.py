"""EXP-053: capacity — does a higher concurrent-position cap (K) / open-risk cap speed the candidate up?
Risk per trade stays 0.5 %. 2019-2026, weekend flat."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.final_candidate import LEGS9, book  # noqa: E402
from research.run_trend_intraday import scan, setup as td_setup  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402

if __name__ == "__main__":
    m1, legs = all_legs()
    m15_td, day, _ = td_setup()
    s = scan(m15_td, day, "tday", 0.3, 1.0, None, 900); s["hold_min"] = 510; s["leg"] = "tday900"
    legs["tday900"] = s.loc[A:B]
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows = []
    for K in (4, 6, 8, 10):
        r, res, m = book(x, legs, LEGS9 + ["tday900"], True, K)
        rows.append({"K": K, "open_risk%": 0.5 * K, **r})
    print(pd.DataFrame(rows).to_string(index=False))

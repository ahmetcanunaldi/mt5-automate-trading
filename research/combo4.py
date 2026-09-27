"""EXP-052: candidate + trend-day variants at other decision times (15:00, 17:00, 19:00) to add frequency."""
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
    for T in (900, 1020, 1140):
        s = scan(m15_td, day, "tday", 0.3, 1.0, None, T)
        s["hold_min"] = 1425 - T - 15
        s["leg"] = f"tday{T}"
        legs[f"tday{T}"] = s.loc[A:B]
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows = []
    for name, names in (("cand9", LEGS9), ("cand9+tday900", LEGS9 + ["tday900"]),
                        ("cand9+tday1020", LEGS9 + ["tday1020"]), ("cand9+tday1140", LEGS9 + ["tday1140"]),
                        ("cand9+all_tday", LEGS9 + ["tday900", "tday1020", "tday1140"])):
        r, res, m = book(x, legs, names, True, 6)
        t = res.trades
        extra = {k: round(v, 1) for k, v in t.groupby("leg").R.sum().items() if k.startswith("tday")}
        rows.append({"book": name, **r, **extra})
    print(pd.DataFrame(rows).to_string(index=False))

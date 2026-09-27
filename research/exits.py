"""EXP-068: global exit rules on the intraday legs of best12+season0.5 (one rule for all legs, no per-leg tuning):
break-even at +1 R, trailing at 1.0 / 1.5 x the leg's own stop distance, or both."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, metrics  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.combo8 import BEST12, build  # noqa: E402
from research.final_candidate import g  # noqa: E402

INTRADAY = {"tday", "tday900", "lw", "inside", "nr7", "friday", "strong_close", "drift", "season"}

if __name__ == "__main__":
    m1, legs = build()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows = []
    for name, be, tr in (("none", 0, 0), ("BE@1R", 1.0, 0), ("trail 1.0x", 0, 1.0), ("trail 1.5x", 0, 1.5),
                         ("BE@1R+trail1.5x", 1.0, 1.5)):
        parts = []
        for n in BEST12 + ["season"]:
            s = legs[n].copy()
            if n == "season":
                s["risk_mult"] = 0.5
            if n in INTRADAY:
                s["be"] = be * s.sl
                if tr:
                    s["trail"] = tr * s.sl
            parts.append(s)
        res = engine.run(x, pd.concat(parts).sort_index(kind="stable"), g(True, 6))
        m = metrics.summarize(res, name); ch = metrics.challenge_sim(res, max_days=250)
        rows.append({"exit": name, "n": m["trades"], "wr": m["win_rate"], "R_yr": round(res.trades.R.sum() / 7.7, 1),
                     "SR": m["sharpe"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
                     "week_R": m["week_R_mean"], "pass250": ch["pass_rate"], "fail": ch["fail_rate"]})
    print(pd.DataFrame(rows).to_string(index=False))

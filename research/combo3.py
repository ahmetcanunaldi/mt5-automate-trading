"""EXP-049: candidate (EXP-047) + Monday big-gap fade + post-holiday long, 2019-2026, weekend flat, K6."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab  # noqa: E402
from research.combo import A, B  # noqa: E402
from research.features import atr  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.final_candidate import LEGS9, book  # noqa: E402
from research.wf_weights import all_legs  # noqa: E402

if __name__ == "__main__":
    m1, legs = all_legs()
    d1 = m1.resample("1D").agg(AGG).dropna(subset=["open"]); d1 = d1[d1.index.dayofweek < 5]
    ad = atr(d1, 14).shift(1)
    c = m1["close"]
    days = d1.loc[A:B].index
    # Monday gap fade: gap from Friday's last close to Monday 01:05
    mon = days[days.dayofweek == 0]
    p_fri = c.asof(mon - pd.Timedelta(minutes=1)).to_numpy()
    p_mon = c.asof(mon + pd.Timedelta(minutes=65)).to_numpy()
    gap = np.log(p_mon / p_fri) * 1e4
    big = np.abs(gap) > 30
    legs["mon_gap"] = pd.DataFrame({"dir": -np.sign(gap[big]).astype(int), "sl": 1.5 * ad.reindex(mon[big]).to_numpy(),
                                    "tp": 1e6, "hold_min": 1330, "be": 0.0, "trail": 0.0, "leg": "mon_gap"},
                                   index=mon[big] + pd.Timedelta(minutes=66)).dropna()
    # post-holiday: first trading day after a weekday market holiday
    alld = d1.index
    prev_gap = pd.Series(alld, index=alld) - pd.Series(alld, index=alld).shift(1)
    post = alld[(prev_gap > pd.Timedelta(days=1)).to_numpy() & (alld.dayofweek != 0)]
    post = post[(post >= A) & (post <= B)]
    legs["post_hol"] = pd.DataFrame({"dir": 1, "sl": 1.5 * ad.reindex(post).to_numpy(), "tp": 1e6, "hold_min": 1330,
                                     "be": 0.0, "trail": 0.0, "leg": "post_hol"}, index=post + pd.Timedelta(minutes=66)).dropna()
    print("mon_gap trades:", len(legs["mon_gap"]), " post_hol:", len(legs["post_hol"]))
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows = []
    for name, names in (("cand9", LEGS9), ("cand9+mon_gap", LEGS9 + ["mon_gap"]),
                        ("cand9+post_hol", LEGS9 + ["post_hol"]), ("cand11", LEGS9 + ["mon_gap", "post_hol"])):
        r, res, m = book(x, legs, names, True, 6)
        rows.append({"book": name, **r})
        t = res.trades
        if "leg" in t:
            print(name, t.groupby("leg").R.agg(["count", "sum"]).round(1).T.to_dict())
    print(pd.DataFrame(rows).to_string(index=False))

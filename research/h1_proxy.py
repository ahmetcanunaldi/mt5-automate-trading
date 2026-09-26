"""EXP-008: regime robustness with H1 proxies of the portfolio legs (pre-M15 history).
NOTE: broker "H1" history before 2018 is actually one bar per day -> only 2018+ is usable.
H1 ATR ~ 2x M15 ATR, so ATR multipliers are halved; Donchian 8 H1 bars ~ 32 M15 bars.
Execution on H1 bars (very pessimistic intrabar: SL first)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, features, lab, metrics  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

if __name__ == "__main__":
    h1 = lab.load("H1")
    df = features.base_frame(h1, h1, bar_minutes=60)
    legs = {
        "donchian_h1": FAMILIES["donchian"](df, n=8, sl_atr=1.0, rr=3.0, hold_min=480, start=900, end=1320,
                                            trend="with", per_day=2, min_width_atr=2.0),
        "pdb_long_h1": FAMILIES["prev_day_break"](df, start=120, end=900, sl_atr=1.0, rr=3.0, trail_atr=1.0,
                                                  hold_min=60, trend="long_only", buffer_atr=0.15),
        "pdb_trend_h1": FAMILIES["prev_day_break"](df, start=120, end=900, sl_atr=1.0, rr=3.0, trail_atr=1.0,
                                                   hold_min=60, trend="with", buffer_atr=0.15),
    }
    news = calendar_news.load_news_server_times()
    for a, b in [("2010-01-01", "2013-12-31"), ("2014-01-01", "2017-12-31"), ("2018-01-01", "2021-12-31"),
                 ("2022-01-01", "2024-11-30"), ("2024-12-01", "2026-09-26")]:
        bars = h1.loc[a:b]
        blk, flt = calendar_news.blackout_masks(bars.index, 60, news, 30, 30, 10)
        if a < "2018":
            blk[:] = False; flt[:] = False          # calendar export starts 2017-12
        x = engine.prepare_exec(bars, 60, blk, flt)
        for name, s in legs.items():
            r = engine.run(x, s.loc[a:b])
            m = metrics.summarize(r, f"{name} {a[:4]}-{b[:4]}")
            print(f"{m['label']:<28} n={m['trades']:>4} PF={m['profit_factor']:.2f} SR={m['sharpe']:>5.2f} "
                  f"net={m['net_pct']:>6.1f}% DD={m['max_total_dd_pct']:.1f}%", flush=True)

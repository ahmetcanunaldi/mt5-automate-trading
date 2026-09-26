"""EXP-026: session AMD models on M15, 2019-2026, $100k account, M1 execution. Full grid reported per year."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.fresh_era import AGG  # noqa: E402
from research.strategies.amd import amd_signals  # noqa: E402

A, B = "2019-01-01", "2026-09-26"
G100 = engine.Guards(initial_balance=100_000, max_trades_day=4, max_positions=1)

if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    rows, keep = [], {}
    grid = itertools.product(["asia_london", "london_ny"], [0.0, 0.25, 0.5], [2, 4, 8], [None, 0.6],
                             ["range", "mid", "rr"])
    for model, sw, cb, mr, tpm in grid:
        s = amd_signals(m15, model, sweep_atr=sw, confirm_bars=cb, max_rng_atr=mr, tp_mode=tpm).loc[A:B]
        name = f"{model}|sw{sw}|cb{cb}|rng{mr}|{tpm}"
        if len(s) < 30:
            continue
        res = engine.run(x, s, G100)
        t = res.trades
        m = metrics.summarize(res, name)
        yr = t.groupby(t.entry_time.dt.year)["R"].mean().round(3)
        rows.append({"cfg": name, "n": m["trades"], "wr": m["win_rate"], "avgR": m["avg_R"], "PF": m["profit_factor"],
                     "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                     "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "yrs_pos": int((yr > 0).sum()),
                     "n_yrs": len(yr), **{f"R{y}": v for y, v in yr.items()}})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("t", ascending=False)
    (lab.REPORTS / "EXP-026").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-026" / "summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(T.head(20).to_string(index=False))
    print(f"\nconfigs: {len(T)}, avgR>0: {(T.avgR > 0).mean():.2f}, median avgR {T.avgR.median():.3f}, "
          f"configs positive in >=6 years: {(T.yrs_pos >= 6).sum()}")
    top = list(T.cfg.head(3))
    lab.save_experiment("EXP-026", {"account": 100_000, "grid": "model x sweep x confirm x range filter x tp"},
                        {c: keep[c][0] for c in top}, {c.replace("|", "_"): keep[c][1] for c in top})

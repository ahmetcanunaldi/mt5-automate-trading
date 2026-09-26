"""EXP-017: TomTrades CBR Asia-reversal model on the full tick era (all variants reported, no tuning)."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.engine_limit import run_orders  # noqa: E402
from research.strategies.cbr import cbr_orders  # noqa: E402

lab.PERIODS["TICK"] = ("2024-12-04", "2026-09-26", "M1")


def exec_tick():
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1T.parquet")
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(m1.index, 1, news, 30, 30, 10)
    return m1, engine.prepare_exec(m1, 1, blk, flt)


if __name__ == "__main__":
    m1, x = exec_tick()
    usdx = pd.read_parquet(lab.DATA / "USDX_M1T.parquet")["close"]
    eur = pd.read_parquet(lab.DATA / "EURUSD_M1T.parquet")["close"]
    g = engine.Guards(max_trades_day=5)
    rows, keep = [], {}
    grid = itertools.product([1, 3], ["usdx", "eurusd", "none"], [True, False], ["1.5R", "ext50"])
    for bias_h, dm, sw, tpm in grid:
        dxy = {"usdx": usdx, "eurusd": eur, "none": None}[dm]
        od = cbr_orders(m1, dxy, bias_h=bias_h, dxy_mode=dm, sweep=sw, tp_mode=tpm)
        name = f"b{bias_h}_{dm}_sw{int(sw)}_{tpm}"
        if len(od) == 0:
            continue
        res = run_orders(x, od, g)
        m = metrics.summarize(res, name)
        m.update(metrics.monte_carlo_dd(res))
        ch = metrics.challenge_sim(res, max_days=120)
        rows.append({"cfg": name, "orders": len(od), "fills": m["trades"], "fill%": round(m["trades"] / len(od), 2),
                     "wr": m["win_rate"], "avgR": m["avg_R"], "PF": m["profit_factor"], "SR": m["sharpe"],
                     "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
                     "med_stop$": round(od.stop_usd.median(), 2), "pass120": ch["pass_rate"]})
        keep[name] = (m, res)
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("SR", ascending=False)
    print(T.to_string(index=False))
    top = list(T.cfg.head(3))
    lab.save_experiment("EXP-017", {"model": "TomTrades CBR", "grid": "bias_h x dxy x sweep x tp"},
                        {k: keep[k][0] for k in top}, {k: keep[k][1] for k in top})
    T.to_csv(lab.REPORTS / "EXP-017" / "summary.csv", index=False)

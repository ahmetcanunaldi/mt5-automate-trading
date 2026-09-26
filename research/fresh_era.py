"""EXP-020: pre-registered candidates A/B (EXP-013, unchanged params) on the untouched 2018..2022-07 era.
Signals from M15/H1 rebuilt from the tester M1 export; execution on M1."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, features, lab, metrics  # noqa: E402
from research.aggressive import chall  # noqa: E402
from research.oos import CANDS  # noqa: E402
from research.portfolio import leg_signals  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum", "spread": "mean"}

if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet")
    m15 = m1.resample("15min").agg(AGG).dropna(subset=["open"])
    h1 = m1.resample("1h").agg(AGG).dropna(subset=["open"])
    df = features.base_frame(m15, h1)
    a, b = "2019-01-01", "2022-07-04"              # 2018 used as indicator warm-up
    ex = m1.loc[a:b]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    results, objs = {}, {}
    for name, (legs, g) in CANDS.items():
        s = leg_signals(df, legs).loc[a:b]
        res = engine.run(x, s, g)
        m = metrics.summarize(res, f"{name}|FRESH")
        m.update(metrics.monte_carlo_dd(res)); m.update(chall(res)); m["gates"] = metrics.gate_check(m)
        results[f"{name}|FRESH"] = m; objs[f"{name}|FRESH"] = res
        print(lab.fmt(m), "| pass120", m["pass120"], flush=True)
        t = res.trades
        print("   by year R:", t.groupby(t.entry_time.dt.year)["R"].agg(["count", "sum"]).round(1).to_dict())
        print("   by leg R :", t.groupby("leg")["R"].agg(["count", "sum"]).round(1).to_dict())
        print("   long/short R:", t.groupby("dir")["R"].agg(["count", "sum"]).round(1).to_dict())
    lab.save_experiment("EXP-020", {"era": [a, b], "candidates": "EXP-013 A/B unchanged"}, results, objs)

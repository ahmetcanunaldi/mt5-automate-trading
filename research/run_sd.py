"""EXP-018: multi-timeframe supply/demand zones, first-touch limit entries.
Era A: 2022-07-05..2024-11-30 executed on M15 bars (pessimistic intrabar); Era B: 2024-12-04..2026-09-25 on M1."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab, metrics  # noqa: E402
from research.engine_limit import run_orders  # noqa: E402
from research.strategies.sd_zones import resample, zone_orders  # noqa: E402

ERAS = {"A": ("2022-07-05", "2024-11-30", "M15"), "B": ("2024-12-04", "2026-09-26", "M1")}


def exec_era(era):
    a, b, tf = ERAS[era]
    bars = lab.load("M15").loc[a:b] if tf == "M15" else pd.read_parquet(lab.DATA / "XAUUSD_M1T.parquet").loc[a:b]
    bm = 15 if tf == "M15" else 1
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(bars.index, bm, news, 30, 30, 10)
    return engine.prepare_exec(bars, bm, blk, flt)


if __name__ == "__main__":
    h1 = lab.load("H1").loc["2018":]
    m15 = lab.load("M15")
    tf_bars = {"M15": m15, "H1": h1, "H4": resample(h1, "H4"), "D1": resample(h1, "D1")}
    X = {e: exec_era(e) for e in ERAS}
    g = engine.Guards(max_trades_day=5, max_positions=2, max_open_risk_pct=1.0)
    rows, objs = [], {}
    for tf, dep, rr, trend in itertools.product(["M15", "H1", "H4", "D1"], [1.5, 2.0], [2.0, 3.0], ["none", "with"]):
        age = {"M15": 48, "H1": 120, "H4": 240, "D1": 480}[tf]
        od = zone_orders(tf_bars[tf], tf, h1, dep_atr=dep, rr=rr, trend=trend, max_age_h=age)
        row = {"tf": tf, "dep": dep, "rr": rr, "trend": trend}
        for e, (a, b, _) in ERAS.items():
            o = od[(od.act_time >= a) & (od.act_time < b)]
            res = run_orders(X[e], o, g)
            m = metrics.summarize(res, "")
            t = res.trades.R
            row.update({f"{e}_n": m["trades"], f"{e}_avgR": m["avg_R"], f"{e}_PF": m["profit_factor"],
                        f"{e}_SR": m["sharpe"], f"{e}_net": m["net_pct"], f"{e}_DD": m["max_total_dd_pct"],
                        f"{e}_t": round(t.mean() / t.std() * len(t) ** 0.5, 2) if len(t) > 5 else None})
            objs[(tf, dep, rr, trend, e)] = (m, res)
        rows.append(row)
        print(row, flush=True)
    T = pd.DataFrame(rows)
    T["min_SR"] = T[["A_SR", "B_SR"]].min(axis=1)
    T = T.sort_values("min_SR", ascending=False)
    (lab.REPORTS / "EXP-018").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-018" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    best = T.iloc[0]
    k = (best.tf, best.dep, best.rr, best.trend)
    lab.save_experiment("EXP-018", {"best": list(map(str, k))},
                        {f"{k}|{e}": objs[(*k, e)][0] for e in ERAS}, {f"{'_'.join(map(str, k))}_{e}": objs[(*k, e)][1] for e in ERAS})

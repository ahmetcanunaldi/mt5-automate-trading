"""EXP-017b: relaxed CBR (shorter extension, larger pullback tolerance, 2nd+3rd Asia hour), 1.5R only."""
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.engine_limit import run_orders  # noqa: E402
from research.run_cbr import exec_tick  # noqa: E402
from research.strategies import cbr  # noqa: E402

if __name__ == "__main__":
    m1, x = exec_tick()
    usdx = pd.read_parquet(lab.DATA / "USDX_M1T.parquet")["close"]
    eur = pd.read_parquet(lab.DATA / "EURUSD_M1T.parquet")["close"]
    g = engine.Guards(max_trades_day=5)
    rows = []
    orig_starts = cbr.session_starts
    for hours, ext_min, pb, (dm, dxy), rr in itertools.product([1, 2], [10, 15, 20], [0.35, 0.5],
                                                               [("usdx", usdx), ("eurusd", eur), ("none", None)],
                                                               [1.5, 2.0]):
        # search window = 2nd hour (hours=1) or 2nd+3rd hour (hours=2): shift session end by patching max window
        od = []
        base = cbr.cbr_orders(m1, dxy, bias_h=3, dxy_mode=dm, sweep=False, tp_mode="1.5R", ext_min=ext_min,
                              pb_frac=pb, rr=rr)
        od.append(base)
        if hours == 2:
            cbr.session_starts = lambda idx: orig_starts(idx) + pd.Timedelta(hours=1)
            od.append(cbr.cbr_orders(m1, dxy, bias_h=3, dxy_mode=dm, sweep=False, tp_mode="1.5R",
                                     ext_min=ext_min, pb_frac=pb, rr=rr))
            cbr.session_starts = orig_starts
        od = pd.concat(od, ignore_index=True)
        od["group"] = od["act_time"].dt.normalize().astype("int64") // 10**9
        if len(od) == 0:
            continue
        res = run_orders(x, od, g)
        m = metrics.summarize(res, "")
        rows.append({"hours": hours, "ext_min": ext_min, "pb": pb, "dxy": dm, "rr": rr, "fills": m["trades"],
                     "wr": m["win_rate"], "avgR": m["avg_R"], "PF": m["profit_factor"], "SR": m["sharpe"],
                     "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                     "t": round(res.trades.R.mean() / res.trades.R.std() * len(res.trades) ** 0.5, 2)})
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows).sort_values("SR", ascending=False)
    (lab.REPORTS / "EXP-017").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-017" / "relaxed.csv", index=False)
    print(T.head(15).to_string(index=False))
    print("share of configs with avgR>0:", round((T.avgR > 0).mean(), 2), " median avgR:", T.avgR.median())

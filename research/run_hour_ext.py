"""EXP-019: TomTrades hourly-extension mean reversion (video wsXu2Xr1nQc) on the tick era, full grid reported."""
import itertools
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.engine_limit import run_orders  # noqa: E402
from research.run_cbr import exec_tick  # noqa: E402
from research.strategies.hour_ext import hour_ext_orders  # noqa: E402

SUB = [("2024-12-04", "2025-06-30"), ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-09-26")]

if __name__ == "__main__":
    m1, x = exec_tick()
    g = engine.Guards(max_trades_day=8)
    rows, keep = [], {}
    grid = itertools.product([0.5, 0.75, 1.0], ["aggr", "shift"], [True, False], [False, True],
                             [("AsiaLon", ((2, 15),)), ("All", ((1, 22),))])
    t0 = time.time()
    for k, ent, sw, rf, (sn, sess) in grid:
        od = hour_ext_orders(m1, k=k, entry=ent, sweep=sw, range_filter=rf, sessions=sess)
        name = f"k{k}_{ent}_sw{int(sw)}_rf{int(rf)}_{sn}"
        if len(od) < 10:
            continue
        res = run_orders(x, od, g)
        tr = res.trades
        m = metrics.summarize(res, name)
        m.update(metrics.monte_carlo_dd(res))
        row = {"cfg": name, "n": m["trades"], "wr": m["win_rate"], "avgR": m["avg_R"], "PF": m["profit_factor"],
               "SR": m["sharpe"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
               "t": round(tr.R.mean() / tr.R.std() * np.sqrt(len(tr)), 2) if len(tr) > 5 else np.nan}
        for a, b in SUB:
            s = tr[(tr.entry_time >= a) & (tr.entry_time <= b)]
            row[f"R_{a[:7]}"] = round(s.R.mean(), 3) if len(s) else np.nan
        rows.append(row); keep[name] = (m, res)
        print(f"[{time.time()-t0:.0f}s]", row, flush=True)
    T = pd.DataFrame(rows).sort_values("SR", ascending=False)
    (lab.REPORTS / "EXP-019").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-019" / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    print("configs with avgR>0:", round((T.avgR > 0).mean(), 2), "| median avgR", T.avgR.median(),
          "| best t", T.t.max())
    top = list(T.cfg.head(3))
    lab.save_experiment("EXP-019", {"source": "https://youtu.be/wsXu2Xr1nQc", "grid": "k x entry x sweep x range x session"},
                        {c: keep[c][0] for c in top}, {c: keep[c][1] for c in top})

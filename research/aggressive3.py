"""EXP-011: expectancy levers on the aggressive set: hold to EOD, volatility-regime filter (ATR ratio)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import features, lab  # noqa: E402
from research.aggressive import chall  # noqa: E402
from research.aggressive2 import LEGS  # noqa: E402
from research.engine import Guards  # noqa: E402
from research.portfolio import leg_signals  # noqa: E402

SET = ["ny_orb", "donchian_us", "pdb_long", "donchian_48", "ny_orb60"]
G = Guards(max_positions=4, max_open_risk_pct=2.0, max_trades_day=10)

if __name__ == "__main__":
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    atr_ratio = (df["atr"] / df["atr_slow"]).set_axis(df["close_time"].values)
    base = leg_signals(df, SET)
    variants = {"base": base}
    v = base.copy(); v.loc[v.leg != "pdb_long", "hold_min"] = 1440; variants["hold_EOD"] = v
    for thr in (0.8, 1.0, 1.2):
        r = atr_ratio.reindex(base.index)
        variants[f"atr_ratio>={thr}"] = base[(r >= thr).to_numpy()]
    r = atr_ratio.reindex(base.index)
    variants["atr_ratio<1.0"] = base[(r < 1.0).to_numpy()]
    rows = []
    for vn, s in variants.items():
        for per in ("DEV", "VAL"):
            m, res = lab.evaluate(s, per, guards=G, label=vn, with_challenge=False)
            rows.append({"variant": vn, "period": per, "n": m["trades"], "SR": m["sharpe"], "PF": m["profit_factor"],
                         "avgR": m["avg_R"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                         "dDD%": m["max_daily_dd_pct"], **chall(res)})
    t = pd.DataFrame(rows)
    t.to_csv(lab.REPORTS / "EXP-011_grid.csv", index=False)
    print(t.to_string())

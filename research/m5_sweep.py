"""EXP-012: M5-signal versions of the US-session trend breakout (tick-era only), random search on TDEV, check TVAL."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import features, lab  # noqa: E402
from research.aggressive import chall  # noqa: E402
from research.engine import Guards  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

SPACE = {
    "donchian": {"n": [24, 48, 72, 96], "sl_atr": [1.5, 2.0, 3.0, 4.0], "start": [600, 900, 990],
                 "end": [1200, 1320], "trend": ["with"], "per_day": [1, 2, 3], "min_width_atr": [0.0, 4.0, 8.0],
                 "rr": [2.0, 3.0], "trail_atr": [0.0, 4.0, 6.0], "hold_min": [240, 480]},
    "prev_day_break": {"start": [120, 600, 900], "end": [900, 1200], "sl_atr": [2.0, 3.0, 4.0],
                       "trend": ["with", "long_only"], "buffer_atr": [0.0, 0.5], "rr": [2.0, 3.0],
                       "trail_atr": [0.0, 4.0], "hold_min": [120, 240]},
}
G = Guards(max_positions=4, max_open_risk_pct=2.0, max_trades_day=10)

if __name__ == "__main__":
    rng = np.random.default_rng(12)
    df = features.base_frame(lab.signal_bars("M5"), lab.signal_bars("H1"), bar_minutes=5)
    rows = []
    for fam, sp in SPACE.items():
        for _ in range(120):
            p = {k: v[rng.integers(len(v))] for k, v in sp.items()}
            s = FAMILIES[fam](df, **p)
            m, _ = lab.evaluate(s, "TDEV", guards=G, label=fam, with_challenge=False)
            rows.append({"family": fam, "params": json.dumps(p), "tdev_n": m["trades"], "tdev_SR": m["sharpe"],
                         "tdev_net": m["net_pct"], "tdev_PF": m["profit_factor"]})
    t = pd.DataFrame(rows).sort_values("tdev_SR", ascending=False)
    t = t[t.tdev_n >= 60]
    for i in t.index[:12]:
        p = json.loads(t.at[i, "params"])
        m, _ = lab.evaluate(FAMILIES[t.at[i, "family"]](df, **p), "TVAL", guards=G, with_challenge=False)
        t.at[i, "tval_n"] = m["trades"]; t.at[i, "tval_SR"] = m["sharpe"]; t.at[i, "tval_net"] = m["net_pct"]
    t.to_csv(lab.REPORTS / "EXP-012_m5_sweep.csv", index=False)
    print(t.head(12).to_string())

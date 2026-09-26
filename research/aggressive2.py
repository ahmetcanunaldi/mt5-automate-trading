"""EXP-010: add US-session trend variants on top of base3; K3/K4 concurrency."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import features, lab  # noqa: E402
from research.aggressive import LEGS, chall  # noqa: E402
from research.engine import Guards  # noqa: E402
from research.portfolio import leg_signals  # noqa: E402

LEGS.update({
    "ny_orb60": ("ny_orb", {"or_start": 990, "or_bars": 4, "end": 1260, "sl_atr": 2.0, "rr": 2.0, "trail_atr": 0.0,
                            "be_r": 0.0, "hold_min": 480, "trend": "with", "min_rng_atr": 0.5, "max_rng_atr": 8.0}),
    "donchian_16": ("donchian", {"n": 16, "sl_atr": 2.0, "rr": 3.0, "trail_atr": 3.0, "be_r": 0.0, "hold_min": 480,
                                 "start": 1020, "end": 1320, "trend": "with", "per_day": 2, "min_width_atr": 4.0}),
    "pdb_trend": ("prev_day_break", {"start": 120, "end": 1200, "sl_atr": 2.0, "rr": 3.0, "trail_atr": 2.0,
                                     "be_r": 0.0, "hold_min": 120, "trend": "with", "buffer_atr": 0.3}),
})

if __name__ == "__main__":
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    base = ["ny_orb", "donchian_us", "pdb_long"]
    leg_sets = {"base3": base, "+don48": base + ["donchian_48"], "+orb60": base + ["ny_orb60"],
                "+don16": base + ["donchian_16"], "+pdbT": base + ["pdb_trend"],
                "+don48+orb60": base + ["donchian_48", "ny_orb60"],
                "+don48+orb60+don16": base + ["donchian_48", "ny_orb60", "donchian_16"]}
    guard_sets = {"K3_r1.5": Guards(max_positions=3, max_open_risk_pct=1.5, max_trades_day=8),
                  "K4_r2.0": Guards(max_positions=4, max_open_risk_pct=2.0, max_trades_day=10)}
    rows = []
    for ln, legs in leg_sets.items():
        s = leg_signals(df, legs)
        for gn, g in guard_sets.items():
            for per in ("DEV", "VAL"):
                m, res = lab.evaluate(s, per, guards=g, label=f"{ln}|{gn}", with_challenge=False)
                rows.append({"legs": ln, "guards": gn, "period": per, "n": m["trades"], "SR": m["sharpe"],
                             "PF": m["profit_factor"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                             "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"], **chall(res)})
    t = pd.DataFrame(rows)
    t.to_csv(lab.REPORTS / "EXP-010_grid.csv", index=False)
    print(t.to_string())

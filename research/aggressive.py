"""EXP-009: aggressiveness via concurrency (same-direction only), more legs, guard settings.
Primary objective: challenge pass probability (P1 8 % + P2 5 %) within 60 / 120 trading days, fail rate."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import features, lab, metrics  # noqa: E402
from research.engine import Guards  # noqa: E402
from research.portfolio import LEGS, leg_signals  # noqa: E402

LEGS.update({
    "london_trend": ("asia_breakout", {"start": 600, "end": 900, "sl_atr": 2.0, "min_rng_atr": 1.0, "max_rng_atr": 6.0,
                                        "trend": "with", "rr": 2.0, "trail_atr": 0.0, "be_r": 0.0, "hold_min": 480}),
    "pullback_us": ("ema_pullback", {"fast": 20, "sl_atr": 2.0, "start": 900, "end": 1260, "touch_atr": 0.3,
                                     "per_day": 2, "trend": "with", "rr": 2.0, "trail_atr": 0.0, "be_r": 0.0,
                                     "hold_min": 480}),
    "donchian_48": ("donchian", {"n": 48, "sl_atr": 2.0, "rr": 3.0, "trail_atr": 0.0, "be_r": 0.0, "hold_min": 480,
                                 "start": 900, "end": 1320, "trend": "with", "per_day": 2, "min_width_atr": 4.0}),
})


def chall(res):
    out = {}
    for md in (60, 120):
        c = metrics.challenge_sim(res, max_days=md)
        out[f"pass{md}"] = c["pass_rate"]; out[f"fail{md}"] = c["fail_rate"]
        out[f"med{md}"] = c["median_days_to_pass"]
    return out


if __name__ == "__main__":
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    leg_sets = {
        "base3": ["ny_orb", "donchian_us", "pdb_long"],
        "base3+pb": ["ny_orb", "donchian_us", "pdb_long", "pullback_us"],
        "base3+lon": ["ny_orb", "donchian_us", "pdb_long", "london_trend"],
        "all6": ["ny_orb", "donchian_us", "pdb_long", "london_trend", "pullback_us", "donchian_48"],
    }
    guard_sets = {
        "K1": Guards(),
        "K2_r1.0": Guards(max_positions=2, max_open_risk_pct=1.0, max_trades_day=8),
        "K3_r1.5": Guards(max_positions=3, max_open_risk_pct=1.5, max_trades_day=8),
    }
    rows = []
    for ln, legs in leg_sets.items():
        s = leg_signals(df, legs)
        for gn, g in guard_sets.items():
            for per in ("DEV", "VAL"):
                m, res = lab.evaluate(s, per, guards=g, label=f"{ln}|{gn}", with_challenge=False)
                rows.append({"legs": ln, "guards": gn, "period": per, "n": m["trades"], "SR": m["sharpe"],
                             "PF": m["profit_factor"], "net%": m["net_pct"], "DD%": m["max_total_dd_pct"],
                             "dDD%": m["max_daily_dd_pct"], "mc95": m["mc_dd95_pct"], **chall(res)})
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(lab.REPORTS / "EXP-009_grid.csv", index=False)
    print(t.to_string())

"""EXP-005: combine robust rule candidates into one single-position portfolio."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import features, lab  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

# central (plateau) parameters, NOT the sweep peaks
LEGS = {
    "ny_orb": ("ny_orb", {"or_start": 990, "or_bars": 2, "end": 1200, "sl_atr": 2.0, "rr": 2.0, "trail_atr": 0.0,
                          "be_r": 0.0, "hold_min": 480, "trend": "with", "min_rng_atr": 0.5, "max_rng_atr": 5.0}),
    "donchian_us": ("donchian", {"n": 32, "sl_atr": 2.0, "rr": 3.0, "trail_atr": 0.0, "be_r": 0.0, "hold_min": 480,
                                 "start": 900, "end": 1320, "trend": "with", "per_day": 2, "min_width_atr": 4.0}),
    "pdb_trend": ("prev_day_break", {"start": 120, "end": 900, "sl_atr": 1.5, "rr": 2.0, "trail_atr": 2.0,
                                     "be_r": 0.0, "hold_min": 120, "trend": "with", "buffer_atr": 0.3}),
    "pdb_long": ("prev_day_break", {"start": 120, "end": 900, "sl_atr": 2.0, "rr": 3.0, "trail_atr": 2.0,
                                    "be_r": 0.0, "hold_min": 60, "trend": "long_only", "buffer_atr": 0.3}),
}


def leg_signals(df, names):
    parts = []
    for n in names:
        fam, p = LEGS[n]
        s = FAMILIES[fam](df, **p)
        s["leg"] = n
        parts.append(s)
    s = pd.concat(parts).sort_index(kind="stable")  # leg order = priority on simultaneous signals (EA mirrors it)
    return s[~s.index.duplicated(keep="first")]


if __name__ == "__main__":
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    combos = [["ny_orb"], ["donchian_us"], ["pdb_trend"], ["ny_orb", "donchian_us"],
              ["ny_orb", "donchian_us", "pdb_trend"], ["ny_orb", "donchian_us", "pdb_long"]]
    results, objs = {}, {}
    for c in combos:
        s = leg_signals(df, c)
        name = "+".join(c)
        for per in ("DEV", "VAL"):
            m, res = lab.evaluate(s, per, label=name)
            results[f"{name}|{per}"] = m
            objs[f"{name}|{per}"] = res
            print(lab.fmt(m), flush=True)
    best = "ny_orb+donchian_us+pdb_long"
    lab.save_experiment("EXP-005", {"legs": LEGS, "combos": combos}, results,
                        {k: v for k, v in objs.items() if k.startswith(best + "|")})
    for per in ("DEV", "VAL"):
        t = objs[f"{best}|{per}"].trades
        print(per, "long/short:", t.groupby("dir")["R"].agg(["count", "mean", "sum"]).round(2).to_dict())
        print(per, "by year:", t.groupby(t.entry_time.dt.year)["R"].agg(["count", "sum"]).round(1).to_dict())

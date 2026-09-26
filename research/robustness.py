"""One-at-a-time parameter sensitivity around a candidate (plateau check) on DEV and VAL."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import features, lab  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402
from research.sweep import EXIT_SPACE, SPACES  # noqa: E402

_df = None


def frame():
    global _df
    if _df is None:
        _df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    return _df


def sensitivity(fam, base: dict, extra_values: dict | None = None):
    df = frame()
    fn = FAMILIES[fam]
    space = {**SPACES[fam], **EXIT_SPACE, **(extra_values or {})}
    rows = []
    for k, vals in space.items():
        for v in sorted(set(vals), key=lambda z: (z is None, str(z))):
            p = dict(base); p[k] = v
            sig = fn(df, **p)
            md, _ = lab.evaluate(sig, "DEV", label=fam, with_challenge=False)
            mv, _ = lab.evaluate(sig, "VAL", label=fam, with_challenge=False)
            rows.append({"param": k, "value": v, "is_base": base.get(k) == v, "dev_n": md["trades"],
                         "dev_sr": md["sharpe"], "val_n": mv["trades"], "val_sr": mv["sharpe"],
                         "dev_pf": md["profit_factor"], "val_pf": mv["profit_factor"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cands = {
        "ny_orb": {"be_r": 0.0, "end": 1080, "hold_min": 240, "max_rng_atr": 5.0, "min_rng_atr": 1.0, "or_bars": 2,
                   "or_start": 990, "rr": 1.5, "sl_atr": 2.0, "trail_atr": 3.0, "trend": "with"},
        "donchian": {"be_r": 0.0, "end": 1320, "hold_min": 240, "min_width_atr": 4.0, "n": 32, "per_day": 1,
                     "rr": 3.0, "sl_atr": 2.0, "start": 900, "trail_atr": 0.0, "trend": "with"},
        "prev_day_break": {"be_r": 0.0, "buffer_atr": 0.3, "end": 900, "hold_min": 60, "rr": 3.0, "sl_atr": 2.0,
                           "start": 120, "trail_atr": 2.0, "trend": "long_only"},
    }
    out = []
    for fam, base in cands.items():
        s = sensitivity(fam, base)
        s.insert(0, "family", fam)
        out.append(s)
        print(f"\n## {fam}\n", s.round(2).to_string(index=False), flush=True)
    pd.concat(out).to_csv(lab.REPORTS / "EXP-004_sensitivity.csv", index=False)

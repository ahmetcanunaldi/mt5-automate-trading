"""EXP-007: trend-constrained sweep -> greedy single-position portfolio built on DEV only, checked on VAL."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import features, lab  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

SPACE = {
    "donchian": {"n": [16, 24, 32, 48, 64], "sl_atr": [1.5, 2.0, 3.0], "start": [120, 480, 600, 780, 900, 1020],
                 "end": [900, 1080, 1320], "trend": ["with"], "per_day": [1, 2], "min_width_atr": [0.0, 2.0, 4.0]},
    "ny_orb": {"or_start": [990], "or_bars": [1, 2, 3, 4], "end": [1080, 1200, 1320], "sl_atr": [1.5, 2.0, 3.0],
               "trend": ["with"], "min_rng_atr": [0.3, 1.0], "max_rng_atr": [5.0, 8.0]},
    "ema_pullback": {"fast": [20, 34, 50], "sl_atr": [1.5, 2.0, 3.0], "start": [120, 600, 900], "end": [900, 1200, 1320],
                     "touch_atr": [0.0, 0.3], "per_day": [1, 2], "trend": ["with"]},
    "prev_day_break": {"start": [120, 600, 900], "end": [900, 1200, 1320], "sl_atr": [1.5, 2.0, 3.0], "trend": ["with"],
                       "buffer_atr": [0.0, 0.3]},
    "asia_breakout": {"start": [600, 660], "end": [780, 900, 1080], "sl_atr": [1.5, 2.0, 3.0], "min_rng_atr": [0.5, 1.0],
                      "max_rng_atr": [6.0, 10.0], "trend": ["with"]},
}
EXITS = {"rr": [2.0, 3.0], "trail_atr": [0.0, 3.0], "be_r": [0.0], "hold_min": [240, 480]}


def main(n_per=60, seed=3):
    rng = np.random.default_rng(seed)
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    pool = []
    for fam, sp in SPACE.items():
        seen = set()
        for _ in range(n_per):
            p = {k: v[rng.integers(len(v))] for k, v in {**sp, **EXITS}.items()}
            key = json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            s = FAMILIES[fam](df, **p)
            m, _ = lab.evaluate(s, "DEV", label=fam, with_challenge=False)
            if m["trades"] >= 100 and m["sharpe"] >= 0.8:
                s = s.copy(); s["leg"] = f"{fam}:{key}"
                pool.append((m["sharpe"], fam, p, s))
        print(fam, "qualified so far:", len(pool), flush=True)
    pool.sort(key=lambda z: -z[0])
    # greedy forward selection on DEV Sharpe (max 6 legs, must improve by >= 0.05)
    chosen, cur_sr, cur = [], -9, None
    for _ in range(6):
        best = None
        for cand in pool:
            if any(cand[3]["leg"].iloc[0] == c[3]["leg"].iloc[0] for c in chosen):
                continue
            sig = pd.concat([c[3] for c in chosen] + [cand[3]]).sort_index()
            sig = sig[~sig.index.duplicated(keep="first")]
            m, _ = lab.evaluate(sig, "DEV", with_challenge=False)
            if best is None or m["sharpe"] > best[0]:
                best = (m["sharpe"], cand, m)
        if best is None or best[0] < cur_sr + 0.05:
            break
        chosen.append(best[1]); cur_sr = best[0]
        print(f"+ {best[1][1]} {best[1][2]} -> DEV SR {best[0]:.2f} n={best[2]['trades']}", flush=True)
    sig = pd.concat([c[3] for c in chosen]).sort_index()
    sig = sig[~sig.index.duplicated(keep="first")]
    results, objs = {}, {}
    for per in ("DEV", "VAL"):
        m, r = lab.evaluate(sig, per, label="greedy")
        results[per] = m; objs[per] = r
        print(lab.fmt(m), flush=True)
    lab.save_experiment("EXP-007", {"legs": [(c[1], c[2]) for c in chosen], "pool_size": len(pool),
                                    "configs_tried": n_per * len(SPACE)}, results, objs)


if __name__ == "__main__":
    main()

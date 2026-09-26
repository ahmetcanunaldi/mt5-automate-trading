"""Random-search parameter sweeps: fit on DEV, confirm on VAL (OOS stays locked).

usage: python research/sweep.py EXP-003 --n 120 [--families a,b] [--seed 1]
Writes reports/<EXP>/sweep.csv (every config tried, so the number of trials is on record).
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import features, lab, metrics  # noqa: E402
from research.engine import Guards  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

EXIT_SPACE = {
    "rr": [1.0, 1.5, 2.0, 3.0, 5.0],
    "trail_atr": [0.0, 0.0, 1.0, 1.5, 2.0, 3.0],
    "be_r": [0.0, 0.0, 1.0],
    "hold_min": [60, 120, 240, 480],
}
SPACES = {
    "asia_breakout": {"start": [600, 630], "end": [720, 780, 900], "sl_atr": [1.0, 1.5, 2.0, 3.0],
                      "min_rng_atr": [0.5, 1.0, 2.0], "max_rng_atr": [4.0, 6.0, 10.0],
                      "trend": ["none", "with", "against", "long_only"]},
    "ny_orb": {"or_start": [930, 990], "or_bars": [1, 2, 4], "end": [1080, 1200, 1320], "sl_atr": [1.0, 1.5, 2.0, 3.0],
               "trend": ["none", "with", "against", "long_only"], "min_rng_atr": [0.3, 0.5, 1.0],
               "max_rng_atr": [3.0, 5.0, 8.0]},
    "ema_pullback": {"fast": [10, 20, 34, 50], "sl_atr": [1.0, 1.5, 2.0, 3.0], "start": [120, 600, 900],
                     "end": [900, 1200, 1320], "touch_atr": [0.0, 0.3, 0.6], "per_day": [1, 2, 3],
                     "trend": ["with", "long_only"]},
    "donchian": {"n": [8, 16, 32, 48], "sl_atr": [1.0, 1.5, 2.0, 3.0], "start": [120, 600, 900],
                 "end": [900, 1200, 1320], "trend": ["none", "with", "long_only"], "per_day": [1, 2, 3],
                 "min_width_atr": [0.0, 2.0, 4.0]},
    "vwap_reversion": {"dev_atr": [1.5, 2.0, 2.5, 3.0, 4.0], "sl_atr": [0.5, 1.0, 1.5], "start": [120, 600],
                       "end": [900, 1200, 1320], "max_slope": [None, 0.5, 1.0, 2.0]},
    "prev_day_break": {"start": [120, 600, 900], "end": [900, 1200, 1320], "sl_atr": [1.0, 1.5, 2.0, 3.0],
                       "trend": ["none", "with", "long_only"], "buffer_atr": [0.0, 0.1, 0.3]},
}


def sample(space, rng):
    return {k: v[rng.integers(len(v))] for k, v in space.items()}


def score(m):
    if m["trades"] < 80:
        return -9
    return m["sharpe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--families", default=",".join(SPACES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    out = []
    t0 = time.time()
    for fam in a.families.split(","):
        fn = FAMILIES[fam]
        seen = set()
        rows = []
        for _ in range(a.n):
            p = sample(SPACES[fam], rng) | sample(EXIT_SPACE, rng)
            key = json.dumps(p, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            sig = fn(df, **p)
            m, _ = lab.evaluate(sig, "DEV", label=fam, with_challenge=False)
            rows.append({"family": fam, "params": key, **{f"dev_{k}": m[k] for k in
                         ("trades", "win_rate", "profit_factor", "net_pct", "sharpe", "max_total_dd_pct",
                          "max_daily_dd_pct")}, "dev_score": score(m)})
        fr = pd.DataFrame(rows).sort_values("dev_score", ascending=False)
        for i in fr.index[: a.topk]:
            p = json.loads(fr.at[i, "params"])
            m, _ = lab.evaluate(fn(df, **p), "VAL", label=fam, with_challenge=False)
            for k in ("trades", "win_rate", "profit_factor", "net_pct", "sharpe", "max_total_dd_pct", "max_daily_dd_pct"):
                fr.at[i, f"val_{k}"] = m[k]
        out.append(fr)
        best = fr.head(a.topk)
        print(f"\n## {fam}: {len(fr)} configs, {time.time()-t0:.0f}s", flush=True)
        print(best[["dev_trades", "dev_sharpe", "dev_profit_factor", "dev_net_pct", "val_trades", "val_sharpe",
                    "val_profit_factor", "val_net_pct"]].round(2).to_string(), flush=True)
    res = pd.concat(out)
    d = lab.REPORTS / a.exp
    d.mkdir(parents=True, exist_ok=True)
    res.to_csv(d / "sweep.csv", index=False)
    print("\nconfigs tried:", len(res))


if __name__ == "__main__":
    main()

"""EXP-013: one-shot OOS evaluation of the two pre-registered candidates."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from research import features, lab  # noqa: E402
from research.aggressive import chall  # noqa: E402
from research.aggressive2 import LEGS  # noqa: E402,F401
from research.engine import Guards  # noqa: E402
from research.portfolio import leg_signals  # noqa: E402

CANDS = {
    "A_base3_K1": (["ny_orb", "donchian_us", "pdb_long"], Guards()),
    "B_aggr_K4": (["ny_orb", "donchian_us", "pdb_long", "donchian_48", "ny_orb60"],
                  Guards(max_positions=4, max_open_risk_pct=2.0, max_trades_day=10)),
}

if __name__ == "__main__":
    lab.evaluate.oos_unlocked = True
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    results, objs = {}, {}
    for name, (legs, g) in CANDS.items():
        s = leg_signals(df, legs)
        for per in ("DEV", "VAL", "OOS"):
            m, res = lab.evaluate(s, per, guards=g, label=name)
            m.update(chall(res))
            results[f"{name}|{per}"] = m; objs[f"{name}|{per}"] = res
            print(lab.fmt(m), "| pass60", m["pass60"], "pass120", m["pass120"], "med120", m["med120"], flush=True)
            print("   gates:", m["gates"], "| by leg R:", res.trades.groupby("leg")["R"].agg(["count", "sum"]).round(1).to_dict() if "leg" in res.trades else "")
    lab.save_experiment("EXP-013", {"candidates": {k: (v[0], vars(v[1])) for k, v in CANDS.items()}}, results, objs)

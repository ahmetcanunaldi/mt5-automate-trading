"""EXP-001: baseline of every rule family with default parameters on DEV and VAL."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research import features, lab  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

if __name__ == "__main__":
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    results, objs = {}, {}
    for name, fn in FAMILIES.items():
        sig = fn(df)
        for per in ("DEV", "VAL"):
            m, res = lab.evaluate(sig, per, label=name)
            results[f"{name}|{per}"] = m
            print(lab.fmt(m), flush=True)
    lab.save_experiment("EXP-001", {"desc": "rule families, default params", "families": list(FAMILIES)}, results)

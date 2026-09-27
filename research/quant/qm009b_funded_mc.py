"""QM-009b: funded-phase Monte Carlo (block bootstrap of v2 daily R, DEV) with sizing policies; payout at +3 %,
account lost at -8 % from the reset balance or a -3 % day; 2-year horizon."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine_multi, lab, symbols  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.portfolio_v2 import book, guards, load_all  # noqa: E402

A, B = "2019-01-01", "2024-12-31"


def funded(R, fn, n_paths=4000, days=504, block=10, seed=3):
    rng = np.random.default_rng(seed); n = len(R); pays, lost = [], 0
    for _ in range(n_paths):
        eq, k, t = 0.0, 0, 0
        while t < days:
            if t % block == 0:
                s = int(rng.integers(0, n - block))
            d = 0.5 * fn(max(0.0, -eq)) * R[s + t % block]; t += 1
            if d <= -3.0 or eq + d <= -8.0:
                lost += 1; break
            eq += d
            if eq >= 3.0:
                k += 1; eq = 0.0
        pays.append(k)
    return round(np.mean(pays) / 2, 2), round(lost / n_paths, 4)


if __name__ == "__main__":
    ax, xau, idx_legs = load_all()
    sig = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR}).loc[A:B]
    costs = {s: symbols.COSTS[s] for s in ["XAUUSD", "NAS100", "DJ30", "GER40"]}
    res = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0))
    eq = res.daily["end"].loc[A:B]; R = (eq.diff().fillna(eq.iloc[0] - 100_000) / 500.0); R = R[R.index.dayofweek < 5].to_numpy()
    rows = []
    for name, fn in (("constant 0.5 %", lambda dd: 1.0), ("step (current)", lambda dd: 1.0 if dd < 6.5 else 0.5),
                     ("cushion from 2 %", lambda dd: float(np.clip((8 - dd) / 6, 0.1, 1))),
                     ("cushion from 4 %", lambda dd: float(np.clip((8 - dd) / 4, 0.1, 1))),
                     ("constant 0.4 %", lambda dd: 0.8), ("0.4 % + cushion from 2 %", lambda dd: 0.8 * float(np.clip((8 - dd) / 6, 0.1, 1)))):
        pay, lost = funded(R, fn)
        rows.append({"policy": name, "payouts/yr": pay, "P(account lost in 2y)": lost}); print(rows[-1], flush=True)
    T = pd.DataFrame(rows); (lab.REPORTS / "quant" / "QM-009").mkdir(parents=True, exist_ok=True)
    T.to_csv(lab.REPORTS / "quant" / "QM-009" / "funded_mc.csv", index=False)
    print(T.to_string(index=False))

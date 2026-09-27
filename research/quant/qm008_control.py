"""QM-008: optimal risk control for the FundingPips objective (DEV 2019-2024 daily R of algorithm v2).

The challenge is a goal-reaching problem (Browne 1995): reach +8 % (then +5 %) before losing 8 % (our floor), no
time limit, a 3 % daily loss limit. With a positive edge, smaller bets raise the success probability and lengthen
the time -> a probability / time frontier. Evaluated (a) analytically with a Brownian approximation of v2's daily
P&L, (b) by block-bootstrap Monte Carlo of v2's actual daily R path (fat tails, clustering, daily limit) for:
constant size f, the current step rule (half size below 6.5 % DD), and cushion-proportional (Grossman-Zhou /
CPPI-like) rules f = clip((8 - dd) / (8 - d0), 0.1, 1).
(c) Funded phase: payout every +3 %, account lost at -8 % from the reset balance; expected payouts per year and
P(account lost within 2 years) for constant f."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine_multi, lab, symbols  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.portfolio_v2 import book, guards, load_all  # noqa: E402
from research.quant import portfolio as P  # noqa: E402

A, B = "2019-01-01", "2024-12-31"
OUT = lab.REPORTS / "quant" / "QM-008"


def funded_mc(R, f, n_paths=3000, days=500, block=10, seed=1):
    rng = np.random.default_rng(seed); n = len(R); pays, lost = [], 0
    for _ in range(n_paths):
        eq, k, t = 0.0, 0, 0
        while t < days:
            if t % block == 0:
                s = int(rng.integers(0, n - block))
            d = 0.5 * f * R[s + t % block]; t += 1
            if d <= -3.0 or eq + d <= -8.0:
                lost += 1; break
            eq += d
            if eq >= 3.0:
                k += 1; eq = 0.0
        pays.append(k)
    return np.mean(pays) / (days / 252), lost / n_paths


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ax, xau, idx_legs = load_all()
    sig = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR}).loc[A:B]
    costs = {s: symbols.COSTS[s] for s in ["XAUUSD", "NAS100", "DJ30", "GER40"]}
    res = engine_multi.run(ax, sig, costs, guards(6, 3.0, risk_on_initial=True, total_stop_pct=100.0, total_derisk_pct=100.0))
    eq = res.daily["end"].loc[A:B]
    R = (eq.diff().fillna(eq.iloc[0] - 100_000) / 500.0)
    R = R[R.index.dayofweek < 5].to_numpy()
    mu, sd = R.mean(), R.std()
    print(f"v2 daily R (DEV 2019-24, fixed size): mean {mu:.3f} R, sd {sd:.3f} R, skew {pd.Series(R).skew():.2f}, "
          f"kurt {pd.Series(R).kurt():.2f}, n {len(R)}")
    rows = []
    for f in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
        m, s = 0.5 * f * mu, 0.5 * f * sd
        p1, t1 = P.bm_pass_probability(m, s, 8, 8); p2, t2 = P.bm_pass_probability(m, s, 5, 8)
        mc = P.policy_mc(R, lambda dd, f=f: f)
        pay, lost = funded_mc(R, f)
        rows.append({"policy": f"constant f={f} ({0.5 * f:.2f} % per R)", "BM_p_pass": round(p1 * p2, 3),
                     "BM_days": round(t1 + t2), "MC_p_pass": mc["p_pass"], "MC_p_fail": mc["p_fail"],
                     "MC_med_days": mc["median_days"], "funded_payouts/yr": round(pay, 2), "funded_P(lost 2y)": round(lost, 3)})
        print(rows[-1], flush=True)
    for name, fn in (("step: 1 until DD 6.5 %, then 0.5 (current)", lambda dd: 1.0 if dd < 6.5 else 0.5),
                     ("cushion linear from DD 2 %", lambda dd: float(np.clip((8 - dd) / 6, 0.1, 1))),
                     ("cushion linear from DD 4 %", lambda dd: float(np.clip((8 - dd) / 4, 0.1, 1))),
                     ("cushion linear from DD 0 % (pure CPPI)", lambda dd: float(np.clip((8 - dd) / 8, 0.1, 1)))):
        mc = P.policy_mc(R, fn)
        rows.append({"policy": name, "MC_p_pass": mc["p_pass"], "MC_p_fail": mc["p_fail"], "MC_med_days": mc["median_days"]})
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "frontier.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))

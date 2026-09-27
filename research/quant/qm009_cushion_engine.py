"""QM-009: the cushion-proportional risk rule (QM-008) on algorithm v2 with the real engine, DEV 2019-2024.
Challenge mode (compounding): step rule (current) vs cushion from 2 % / 4 % DD. Funded mode (fixed $500, +3 %
payouts, 35 % consistency, 1.25 % day cap): same rules, plus risk 0.4 %."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402

from research import engine_multi, lab, metrics, symbols  # noqa: E402
from research.combo_idx import PRIOR  # noqa: E402
from research.payout_sim import summarize_payouts  # noqa: E402
from research.portfolio_v2 import book, guards, load_all  # noqa: E402

A, B = "2019-01-01", "2024-12-31"
OUT = lab.REPORTS / "quant" / "QM-009"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ax, xau, idx_legs = load_all()
    sig = book(xau, idx_legs, {"NAS100": PRIOR, "DJ30": PRIOR}).loc[A:B]
    costs = {s: symbols.COSTS[s] for s in ["XAUUSD", "NAS100", "DJ30", "GER40"]}
    rows = []
    for name, kw in (("step (current)", {}), ("cushion from 2 %", {"cushion_start_pct": 2.0}),
                     ("cushion from 4 %", {"cushion_start_pct": 4.0})):
        res = engine_multi.run(ax, sig, costs, guards(6, 3.0, **kw))
        m = metrics.summarize(res); ch = metrics.challenge_sim(res, max_days=250); chu = metrics.challenge_sim(res, max_days=2000)
        row = {"rule": name, "SR": m["sharpe"], "CAGR%": m["cagr_pct"], "DD%": m["max_total_dd_pct"], "dDD%": m["max_daily_dd_pct"],
               "pass250": ch["pass_rate"], "fail250": ch["fail_rate"], "pass_unlim": chu["pass_rate"], "fail_unlim": chu["fail_rate"],
               "med_days": chu["median_days_to_pass"]}
        for rk, rp in (("0.5", 0.5), ("0.4", 0.4)):
            gf = guards(6, 3.0, risk_on_initial=True, payout_pct=3.0, consistency_pct=35.0, day_profit_cap_pct=1.25, **kw)
            gf = dataclasses.replace(gf, risk_pct=rp)
            r, _ = summarize_payouts(engine_multi.run(ax, sig, costs, gf), "")
            row.update({f"funded{rk}_payouts": r["payouts"], f"funded{rk}_$": r.get("withdrawn_$"),
                        f"funded{rk}_med_d": r.get("median_days"), f"funded{rk}_breach": r.get("account_breached")})
        rows.append(row); print(row, flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.T.to_string())

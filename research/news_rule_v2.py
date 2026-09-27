"""EXP-082: best13 under the new news rule (user 2026-09-27 + FundingPips funded rule): no entries -10..+10 min,
positions opened < 5 h before the event flattened 10 min before it, older positions held (FP 5-hour exemption).
Compared with the old rule (entries -30..+30, flatten everything 10 min before)."""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import engine, lab, legcache, symbols  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.final_candidate import g  # noqa: E402
from research.news_scope import book_row  # noqa: E402
from research.payout_sim import A, guards as pguards, summarize_payouts  # noqa: E402
from research import metrics  # noqa: E402

if __name__ == "__main__":
    m1, legs = legcache.load()
    ex = m1.loc[A:B]
    sig = book_signals(legs, A)
    rows, keep = [], {}
    for name, (bef, aft, ex_min) in {"old -30/+30 flatten all": (30, 30, 0),
                                     "new -10/+10 flatten all": (10, 10, 0),
                                     "new -10/+10 + 5h exemption": (10, 10, 290)}.items():
        x = symbols.prepare("XAUUSD", ex, bef, aft, 10)
        gc = dataclasses.replace(g(True, 6), news_exempt_min=ex_min)
        row, res, m = book_row(x, sig, name, gc)
        for cap in (0.0, 1.25):
            r, _ = summarize_payouts(engine.run(x, sig, dataclasses.replace(pguards(3.0, 35.0, cap), news_exempt_min=ex_min)), "")
            mf = metrics.summarize(engine.run(x, sig, dataclasses.replace(pguards(0.0, 100.0, cap), news_exempt_min=ex_min)), "")
            row.update({f"cap{cap}_pay": r["payouts"], f"cap{cap}_med_d": r.get("median_days"),
                        f"cap{cap}_$": r.get("withdrawn_$"), f"cap{cap}_SR": mf["sharpe"], f"cap{cap}_wkR": mf["week_R_mean"]})
        rows.append(row); keep[name] = (m, res)
        print(row, flush=True)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).T.to_string())
    lab.save_experiment("EXP-082", {"rule": "news v2"}, {k: v[0] for k, v in keep.items()},
                        {k.replace(" ", "_").replace("/", "").replace("+", "p"): v[1] for k, v in keep.items()})

"""EXP-027: swing strategy battery on daily bars 2008-2026 ($100k, swing engine with swaps, no total stop for
diagnosis). Consistency judged per era: 2008-12, 2013-18, 2019-22, 2023-26 and per year."""
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.strategies import swing  # noqa: E402

ERAS = [("2008", "2012"), ("2013", "2018"), ("2019", "2022"), ("2023", "2026")]
COSTS = engine.Costs(min_spread_pts=25.0)


def guards(weekend_flat=False, stop=100.0):
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=weekend_flat, first_entry_min=0,
                         last_entry_min=1440, max_trades_day=2, total_stop_pct=stop, total_derisk_pct=stop)


def exec_d1():
    d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet").loc["2008":]
    return d, engine.prepare_exec(d, 1440, costs=COSTS)


def evaluate(x, s, g, name):
    res = engine.run(x, s, g, COSTS)
    t = res.trades
    if len(t) < 20:
        return None, res
    m = metrics.summarize(res, name)
    yr = t.groupby(t.entry_time.dt.year)["R"].sum()
    row = {"cfg": name, "n": len(t), "tr_per_yr": round(len(t) / 18.7, 1), "wr": m["win_rate"], "avgR": m["avg_R"],
           "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "SR": m["sharpe"], "net%": m["net_pct"],
           "DD%": m["max_total_dd_pct"], "yrs_pos": int((yr > 0).sum()), "n_yrs": len(yr)}
    for a, b in ERAS:
        e = t[(t.entry_time >= a) & (t.entry_time < str(int(b) + 1))]
        row[f"R_{a}"] = round(e.R.mean(), 3) if len(e) else np.nan
    return row, res


GRIDS = {
    "donchian": dict(n=[20, 55, 100], sl=[2.0, 3.0], trail=[3.0, 5.0], hold=[20, 60], filt=[None, "ema200"],
                     both=[True, False]),
    "ema_state": dict(fast=[10, 20, 50], slow=[50, 100, 200], sl=[2.0, 3.0], trail=[0.0, 4.0], hold=[1, 5, 20],
                      both=[True, False]),
    "tsmom": dict(look=["ret20", "ret60", "ret120"], sl=[2.0, 3.0], hold=[5], both=[True, False]),
    "week_break": dict(sl=[1.5, 2.5], trail=[2.5, 4.0], hold=[10, 20], filt=[None, "ema100"]),
    "rsi2": dict(lo=[5, 10, 20], hi=[80, 90, 95], sl=[2.0, 3.0], tp=[1.0, 2.0], hold=[3, 5], both=[True, False]),
    "bollinger_mr": dict(sl=[2.0, 3.0], tp=[1.0, 2.0], hold=[3, 5, 10], both=[True, False]),
    "turn_of_month": dict(days_before=[0, 1, 2], days_after=[1, 3, 5], sl=[2.0, 3.0]),
    "weekday": dict(dow=[0, 1, 2, 3, 4], direction=[1, -1], sl=[1.5]),
}

if __name__ == "__main__":
    d, x = exec_d1()
    df = swing.prep(pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet"))
    rows = []
    for fam, grid in GRIDS.items():
        keys = list(grid)
        for vals in itertools.product(*[grid[k] for k in keys]):
            p = dict(zip(keys, vals))
            if fam == "ema_state" and p["fast"] >= p["slow"]:
                continue
            s = swing.LIB[fam](df, **p).loc["2008-02":]
            for wf in (False,):
                row, _ = evaluate(x, s, guards(wf), f"{fam}|{json.dumps(p)}")
                if row:
                    row["family"] = fam; rows.append(row)
        print(fam, "done", flush=True)
    T = pd.DataFrame(rows)
    era = [c for c in T.columns if c.startswith("R_")]
    T["eras_pos"] = (T[era] > 0).sum(axis=1)
    T = T.sort_values(["eras_pos", "t"], ascending=False)
    (lab.REPORTS / "EXP-027").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-027" / "summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(f"configs {len(T)}; all 4 eras positive: {(T.eras_pos == 4).sum()}")
    print(T.head(30)[["cfg", "n", "tr_per_yr", "wr", "avgR", "t", "SR", "net%", "DD%", "yrs_pos", "n_yrs"] + era].to_string(index=False))
    print("\nbest per family:")
    print(T.groupby("family").head(1)[["cfg", "n", "avgR", "t", "SR", "yrs_pos", "n_yrs"] + era].to_string(index=False))

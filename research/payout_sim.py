"""EXP-070: funded-account payout simulation of the best book (best13), 2019-2026.

Rules: $100k, risk 0.5 % of the INITIAL balance, our guards (3 % daily / 8 % total vs initial) + FP limits;
when cycle profit >= payout_pct and the best day of the cycle <= consistency_pct of the cycle profit and no
position is open at the day change, the profit is withdrawn and the balance returns to $100k.
Reports payout count, amounts, intervals, consistency delays, and a chart (account equity + cumulative withdrawals).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, engine, lab  # noqa: E402
from research.best_report import book_signals  # noqa: E402
from research.combo import B  # noqa: E402
from research.combo8 import build  # noqa: E402
from research.plots import C_EQ, C_FP, C_OURS, C_TGT, GRID, INK, INK2, _style  # noqa: E402

A = "2019-01-01"


def guards(payout, cons, cap=0.0):
    return engine.Guards(initial_balance=100_000, intraday=False, weekend_flat=True, last_entry_min=1390,
                         flatten_min=1425, fri_flatten_min=1350, max_trades_day=8, max_positions=6,
                         max_open_risk_pct=3.0, risk_on_initial=True, payout_pct=payout, consistency_pct=cons,
                         day_profit_cap_pct=cap)


def summarize_payouts(res, label):
    p = res.params["payouts"].copy()
    if p.empty:
        return {"case": label, "payouts": 0}, p
    start = res.equity.index[0]
    p["prev"] = p["time"].shift(1).fillna(start)
    p["days"] = (p["time"] - p["prev"]).dt.days
    p["trading_days"] = [int(((res.daily.index > a) & (res.daily.index <= b)).sum()) for a, b in zip(p["prev"], p["time"])]
    p["consistency_wait_d"] = (p["time"] - p["reached"]).dt.days
    yrs = (res.equity.index[-1] - start).days / 365.25
    stopped = res.trades.entry_time.max() < res.equity.index[-1] - pd.Timedelta(days=60)
    out = {"case": label, "payouts": len(p), "per_year": round(len(p) / yrs, 1),
           "withdrawn_$": round(p.amount.sum()), "avg_payout_$": round(p.amount.mean()),
           "median_days": float(p.days.median()), "mean_days": round(p.days.mean(), 1),
           "p75_days": float(p.days.quantile(0.75)), "p90_days": float(p.days.quantile(0.9)), "max_days": int(p.days.max()),
           "median_trading_days": float(p.trading_days.median()),
           "waited_for_consistency_%": round((p.consistency_wait_d > 0).mean() * 100),
           "mean_consistency_wait_d": round(p.consistency_wait_d.mean(), 1),
           "account_breached": bool(stopped)}
    return out, p


def chart(res, p, path, title):
    eq = res.daily["end"]
    wd = pd.Series(0.0, index=eq.index)
    for t, a in zip(p["time"], p["amount"]):
        wd.loc[wd.index >= t.normalize()] += a
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    for a in ax:
        _style(a)
    a = ax[0]
    a.plot(eq.index, eq, color=C_EQ, lw=1.5, label="Account equity (resets to $100k at each payout)")
    a.axhline(103_000, color=C_TGT, lw=1.2, ls="--", label="Payout threshold +3 %")
    a.axhline(92_000, color=C_OURS, lw=1.5, ls="--", label="Our total floor −8 %")
    a.axhline(90_000, color=C_FP, lw=1.5, ls=":", label="FundingPips max loss −10 %")
    for t in p["time"]:
        a.axvline(t, color=C_TGT, lw=0.5, alpha=0.35)
    a.set_ylabel("USD", color=INK2, fontsize=9); a.set_title(title, loc="left", fontsize=10.5, color=INK)
    a.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a = ax[1]
    a.step(wd.index, wd.values, where="post", color=C_TGT, lw=2, label="Cumulative withdrawn (gross, before profit split)")
    a.set_ylabel("USD", color=INK2, fontsize=9)
    a.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    m1, legs = build()
    ex = m1.loc[A:B]
    news = calendar_news.load_news_server_times()
    blk, flt = calendar_news.blackout_masks(ex.index, 1, news, 30, 30, 10)
    x = engine.prepare_exec(ex, 1, blk, flt)
    sig = book_signals(legs, A)
    rows, tables = [], {}
    for label, pay, cons in (("3 % + consistency 35 %", 3.0, 35.0), ("3 % without consistency", 3.0, 100.0),
                             ("2 % + consistency 35 %", 2.0, 35.0)):
        res = engine.run(x, sig, guards(pay, cons))
        r, p = summarize_payouts(res, label)
        rows.append(r); tables[label] = (res, p)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    res, p = tables["3 % + consistency 35 %"]
    d = lab.REPORTS / "EXP-070"; d.mkdir(exist_ok=True)
    p.to_csv(d / "payouts_3pct_cons35.csv", index=False)
    T.to_csv(d / "summary.csv", index=False)
    print("\npayouts per calendar year (3 % + 35 %):", p.groupby(p.time.dt.year).amount.agg(["count", "sum"]).round(0).to_dict())
    print("interval distribution (days):", p.days.describe().round(1).to_dict())
    chart(res, p, d / "payouts_3pct_cons35.png",
          f"EXP-070 best13 funded 2019–26: {len(p)} payouts of +3 %, median every {p.days.median():.0f} days, "
          f"withdrawn ${p.amount.sum():,.0f}")

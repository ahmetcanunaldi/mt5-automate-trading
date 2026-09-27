"""Equity report (one PNG per run) with FundingPips / internal limits drawn the way FP computes them.

Panel 1: equity (day close) + intraday worst equity; per-day DAILY loss floors re-based every server day on that
         day's start reference = max(balance, equity) at 00:00 (FP 5 %, ours 3 %); static TOTAL floors vs the
         initial balance (FP 10 %, ours 8 %); Phase-1 target and P1/P2 pass markers.
Panel 2: daily drawdown vs the day's start reference, with the 3 % / 5 % limits.
Panel 3: total drawdown vs the initial balance (FP static rule), with the 8 % / 10 % limits.
Panel 4: weekly P&L in R (R = 0.5 % of the initial balance) with the +2 R/week target.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C_EQ, C_EQ_LOW, C_OURS, C_FP, C_TGT = "#2a78d6", "#9ec3ee", "#eb6834", "#e34948", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def phase_marks(res, init, t1=8.0, t2=5.0, min_days=3):
    d = res.daily[res.daily.index.dayofweek < 5]
    traded = set(res.trades.entry_time.dt.normalize())
    marks, base, target, days, phase = [], init, t1, 0, 1
    for day, row in d.iterrows():
        if row["min"] <= init * 0.90 or row["min"] <= row["start"] * 0.95:
            marks.append((day, f"FP breach (P{phase})")); break
        days += day in traded
        if row["end"] >= base * (1 + target / 100) and days >= min_days:
            marks.append((day, f"P{phase} pass"))
            if phase == 2:
                break
            phase, base, target, days = 2, row["end"], t2, 0
    return marks


def _style(a):
    a.grid(True, color=GRID, lw=0.6)
    a.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)


def equity_report(res, title, path, guards=None):
    g = res.params["guards"]
    init, rp = g["initial_balance"], g["risk_pct"] / 100.0
    hard = g["daily_hard_pct"] / 100.0
    d = res.daily[res.daily.index.dayofweek < 5].copy()
    d["dd_day"] = ((d["start"] - d["min"]) / d["start"] * 100).clip(lower=0)
    d["dd_tot"] = ((init - d["min"]) / init * 100).clip(lower=0)
    peak = np.maximum.accumulate(np.maximum(d["end"].to_numpy(), init))
    d["dd_peak"] = (peak - d["min"].to_numpy()) / peak * 100
    wk = d["end"].resample("W-FRI").last().dropna()
    wr = (wk - wk.shift(1).fillna(init)) / (init * rp)

    fig, ax = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.1, 1.1, 1.2]})
    for a in ax:
        _style(a)
    a = ax[0]
    a.fill_between(d.index, d["min"], d["end"], color=C_EQ_LOW, lw=0, step="mid", label="Intraday worst equity")
    a.plot(d.index, d["end"], color=C_EQ, lw=2, label="Equity (day close)")
    a.step(d.index, d["start"] * (1 - hard), where="mid", color=C_OURS, lw=0.9, alpha=0.9,
           label=f"Daily floor, ours: day start × (1 − {hard * 100:.0f} %)")
    a.step(d.index, d["start"] * 0.95, where="mid", color=C_FP, lw=0.9, alpha=0.7,
           label="Daily floor, FundingPips: day start × (1 − 5 %)")
    a.axhline(init, color=INK2, lw=0.8)
    a.axhline(init * 1.08, color=C_TGT, lw=1.2, ls="--", label="Phase-1 target +8 %")
    a.axhline(init * (1 - g["total_stop_pct"] / 100), color=C_OURS, lw=1.5, ls="--",
              label=f"Total floor, ours −{g['total_stop_pct']:.0f} % (static)")
    a.axhline(init * 0.90, color=C_FP, lw=1.5, ls="--", label="Total floor, FundingPips −10 % (static)")
    for day, txt in phase_marks(res, init):
        a.axvline(day, color=C_FP if "breach" in txt else C_TGT, lw=1, alpha=0.7)
        a.annotate(txt, (day, a.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", fontsize=8, color=INK)
    a.set_ylabel("USD", color=INK2, fontsize=9)
    a.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a.set_title(title, loc="left", fontsize=10.5, color=INK)
    a = ax[1]
    a.bar(d.index, d["dd_day"], width=0.8, color=C_EQ, label="Daily DD % vs day-start reference")
    a.axhline(hard * 100, color=C_OURS, lw=1.5, ls="--", label=f"Our daily limit {hard * 100:.0f} %")
    a.axhline(5.0, color=C_FP, lw=1.5, ls=":", label="FundingPips daily 5 %")
    a.set_ylim(0, 5.6); a.invert_yaxis(); a.set_ylabel("Daily DD %", color=INK2, fontsize=9)
    a.legend(loc="center right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a = ax[2]
    a.fill_between(d.index, 0, d["dd_tot"], color=C_EQ, lw=0, step="mid", label="Total DD % vs initial (FP static)")
    a.plot(d.index, d["dd_peak"], color=INK2, lw=0.8, label="DD % from peak")
    a.axhline(g["total_stop_pct"], color=C_OURS, lw=1.5, ls="--", label=f"Our limit {g['total_stop_pct']:.0f} %")
    a.axhline(10.0, color=C_FP, lw=1.5, ls=":", label="FundingPips 10 %")
    a.set_ylim(0, 11); a.invert_yaxis(); a.set_ylabel("Total DD %", color=INK2, fontsize=9)
    a.legend(loc="center right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a = ax[3]
    a.bar(wr.index, wr.values, width=5, color=np.where(wr.values >= 2, C_TGT, np.where(wr.values >= 0, C_EQ, C_FP)))
    a.axhline(2.0, color=C_TGT, lw=1.5, ls="--", label=f"Target +2 R/week (share of weeks ≥ 2 R: {(wr >= 2).mean() * 100:.0f} %)")
    a.axhline(wr.mean(), color=INK2, lw=1, ls=":", label=f"Mean {wr.mean():.2f} R/week")
    a.axhline(0, color=INK2, lw=0.6)
    a.set_ylabel("Weekly R", color=INK2, fontsize=9)
    a.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return path

"""Equity-curve report with FundingPips / internal drawdown limits (one PNG per run).

Panel 1: equity (daily close) + intraday worst equity, static total-loss floors, Phase-1 target,
         and challenge phase pass markers (P1 +8 %, P2 +5 %, >= 3 trading days each).
Panel 2: daily drawdown per server day (vs day-start balance) with the 3 % (ours) / 5 % (FP) limits.
Panel 3: total drawdown vs the initial balance (FP's static rule) with the 8 % (ours) / 10 % (FP) limits.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# reference palette (dataviz skill): series blue, our limit orange, FP limit red (critical), target green
C_EQ, C_EQ_LOW, C_OURS, C_FP, C_TGT = "#2a78d6", "#9ec3ee", "#eb6834", "#e34948", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def phase_marks(res, init, t1=8.0, t2=5.0, min_days=3):
    """Walk the daily equity once from the start: P1 pass date, then P2 pass date (P2 re-bases at P1 end)."""
    d = res.daily[res.daily.index.dayofweek < 5]
    traded = set(res.trades.entry_time.dt.normalize())
    marks, base, target, days, phase = [], init, t1, 0, 1
    for day, row in d.iterrows():
        if row["min"] <= base * (1 - 0.10) or row["min"] <= row["start"] * (1 - 0.05):
            marks.append((day, f"FP breach (P{phase})")); break
        days += day in traded
        if row["end"] >= base * (1 + target / 100) and days >= min_days:
            marks.append((day, f"P{phase} pass"))
            if phase == 2:
                break
            phase, base, target, days = 2, row["end"], t2, 0
    return marks


def equity_report(res, title, path, guards=None):
    g = res.params["guards"]
    init = g["initial_balance"]
    d = res.daily[res.daily.index.dayofweek < 5].copy()
    d["dd_day"] = ((d["start"] - d["min"]) / d["start"] * 100).clip(lower=0)
    d["dd_tot"] = ((init - d["min"]) / init * 100).clip(lower=0)
    peak = np.maximum.accumulate(np.maximum(d["end"].to_numpy(), init))
    d["dd_peak"] = (peak - d["min"].to_numpy()) / peak * 100

    fig, ax = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    for a in ax:
        a.grid(True, color=GRID, lw=0.6)
        a.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
    # --- panel 1: equity
    a = ax[0]
    a.fill_between(d.index, d["min"], d["end"], color=C_EQ_LOW, lw=0, step="mid", label="Intraday worst equity")
    a.plot(d.index, d["end"], color=C_EQ, lw=2, label="Equity (day close)")
    a.axhline(init, color=INK2, lw=0.8)
    a.axhline(init * (1 + 0.08), color=C_TGT, lw=1.2, ls="--", label="Phase-1 target +8 %")
    a.axhline(init * (1 - g["total_stop_pct"] / 100), color=C_OURS, lw=1.5, ls="--",
              label=f"Our total limit −{g['total_stop_pct']:.0f} %")
    a.axhline(init * 0.90, color=C_FP, lw=1.5, ls=":", label="FundingPips max loss −10 %")
    for day, txt in phase_marks(res, init):
        col = C_FP if "breach" in txt else C_TGT
        a.axvline(day, color=col, lw=1, alpha=0.7)
        a.annotate(txt, (day, a.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", fontsize=8, color=INK)
    a.set_ylabel("USD", color=INK2, fontsize=9)
    a.legend(loc="upper left", fontsize=8, frameon=False)
    a.set_title(title, loc="left", fontsize=11, color=INK)
    # --- panel 2: daily DD
    a = ax[1]
    a.bar(d.index, d["dd_day"], width=0.8, color=C_EQ, label="Daily DD % (vs day start)")
    a.axhline(g["daily_hard_pct"], color=C_OURS, lw=1.5, ls="--", label=f"Our daily limit {g['daily_hard_pct']:.0f} %")
    a.axhline(5.0, color=C_FP, lw=1.5, ls=":", label="FundingPips daily 5 %")
    a.set_ylim(0, 5.6)
    a.invert_yaxis()
    a.set_ylabel("Daily DD %", color=INK2, fontsize=9)
    a.legend(loc="center right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    # --- panel 3: total DD
    a = ax[2]
    a.fill_between(d.index, 0, d["dd_tot"], color=C_EQ, lw=0, step="mid", label="Total DD % vs initial (FP static)")
    a.plot(d.index, d["dd_peak"], color=INK2, lw=0.8, label="DD % from peak")
    a.axhline(g["total_stop_pct"], color=C_OURS, lw=1.5, ls="--", label=f"Our limit {g['total_stop_pct']:.0f} %")
    a.axhline(10.0, color=C_FP, lw=1.5, ls=":", label="FundingPips 10 %")
    a.set_ylim(0, 11)
    a.invert_yaxis()
    a.set_ylabel("Total DD %", color=INK2, fontsize=9)
    a.legend(loc="center right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return path

"""EXP-016: price-action event study on the tick era (2024-12..2026-09), dynamic barriers.

All conditions are written in DIRECTION-SIGNED feature space, so one rule covers long and short
(e.g. signed state +1 = HH/HL for a long row, LH/LL for a short row).
For each event: count, mean realized R (net of costs) under several barrier schemes, t-stat, and the
mean R in three sub-periods (stability).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

LABELS = ["VOL_1.5_2.25", "VOL_2.0_3.0", "VOL_3.0_4.5", "VOL_2.0_2.0", "STRUCT"]
PERIODS = [("2024-12", "2025-06"), ("2025-07", "2025-12"), ("2026-01", "2026-09")]


def events(D):
    ev = {}
    g = D.groupby("dir", group_keys=False)
    prev = lambda col: g[col].shift(1)  # noqa: E731
    for s in ("mi", "me", "ma"):
        st, bw, ba = D[f"{s}_state"], D[f"{s}_bos_with"], D[f"{s}_bos_against"]
        new_bos = (bw > 0) & (prev(f"{s}_bos_with").fillna(0) <= 0)
        ev[f"{s}:BOS_with_structure"] = new_bos & (st == 1)
        ev[f"{s}:BOS_mixed"] = new_bos & (st == 0)
        ev[f"{s}:CHoCH_reversal"] = new_bos & (st == -1)
        # pullback to the last swing point behind us inside a trend, with a candle in our direction
        ev[f"{s}:pullback_to_HL"] = (st == 1) & (D[f"{s}_dl"].between(0, 0.5)) & (D["body"] > 0)
        # bounce from a multi-touch level behind us
        ev[f"{s}:SR_bounce"] = (D[f"{s}_behind_d"] < 0.3) & (D[f"{s}_behind_t"] >= 2) & (D["body"] > 0.3)
        # breakout of a multi-touch level (level just behind, was ahead one bar ago)
        ev[f"{s}:SR_breakout"] = (D[f"{s}_behind_d"] < 0.5) & (D[f"{s}_behind_t"] >= 2) & (D["body"] > 0.5) & \
                                 (prev(f"{s}_ahead_d") < 0.5)
        ev[f"{s}:fade_breakout"] = (ba > 0) & (prev(f"{s}_bos_against").fillna(0) <= 0) & (st == 1)
    # daily levels
    ev["day:PDH_break"] = (D["pdh_d"] > 0) & (prev("pdh_d") <= 0)            # signed: PDH for long, PDL for short
    ev["day:PDL_sweep_reclaim"] = (D["pdl_d"] > 0) & (prev("pdl_d") <= 0) & (D["dir"] == 1)
    ev["day:PDH_sweep_reject"] = (D["pdh_d"] > 0) & (prev("pdh_d") <= 0) & (D["dir"] == -1)
    ev["h1trend:with"] = D["h1_trend"] > 1
    ev["h1trend:with+me_BOS"] = (D["h1_trend"] > 1) & ev["me:BOS_with_structure"]
    ev["h1trend:with+ma_pullback"] = (D["h1_trend"] > 1) & ev["ma:pullback_to_HL"]
    return ev


def main():
    D = pd.read_parquet(lab.DATA / "ml_dataset_m5.parquet")
    D = D.sort_values(["dir", "close_time"]).reset_index(drop=True)
    ev = events(D)
    rows = []
    for name, m in ev.items():
        m = m.fillna(False)
        sub = D[m]
        if len(sub) < 50:
            continue
        row = {"event": name, "n": len(sub), "per_day": round(len(sub) / 440, 2)}
        for lb in LABELS:
            r = sub[f"R_{lb}"].dropna()
            row[f"{lb}"] = round(r.mean(), 3)
            row[f"t_{lb}"] = round(r.mean() / r.std() * np.sqrt(len(r)), 1)
        for a, b in PERIODS:
            p = sub[(sub.close_time >= a) & (sub.close_time < pd.Timestamp(b) + pd.offsets.MonthEnd(1))]
            row[f"R23_{a}"] = round(p["R_VOL_2.0_3.0"].mean(), 3)
        rows.append(row)
    base = {"event": "ALL bars (baseline)", "n": len(D)}
    for lb in LABELS:
        base[lb] = round(D[f"R_{lb}"].mean(), 3)
    T = pd.DataFrame([base] + rows)
    pd.set_option("display.width", 250)
    cols = ["event", "n", "per_day"] + LABELS + [f"t_{x}" for x in ("VOL_2.0_3.0", "STRUCT")] + [c for c in T.columns if c.startswith("R23_")]
    print(T[cols].sort_values("VOL_2.0_3.0", ascending=False).to_string(index=False))
    (lab.REPORTS / "EXP-016").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-016" / "events.csv", index=False)


if __name__ == "__main__":
    main()

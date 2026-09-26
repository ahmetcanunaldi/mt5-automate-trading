"""EXP-028: intraday seasonality map on M1 2019-2026 (+ D1 2008-2018 for weekday checks).
For every weekday x entry hour (server) x hold (1,2,4,8 h): mean log return (bps) and in ATR_D units, t-stat,
share of years with the same sign, and first-half vs second-half (2019-22 vs 2023-26) agreement.
Multiple testing: 5 x 23 x 4 = 460 tests -> Bonferroni |t| > 3.5 for 5 %."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

if __name__ == "__main__":
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2019":]
    h1 = m1["close"].resample("1h").last().dropna()
    op = m1["open"].resample("1h").first().reindex(h1.index)
    rows = []
    for hold in (1, 2, 4, 8):
        fwd = np.log(h1.shift(-(hold - 1)) / op) * 1e4            # open of hour H -> close of hour H+hold-1
        same_day = h1.index.normalize() == pd.Series(h1.index, index=h1.index).shift(-(hold - 1)).dt.normalize()
        f = pd.DataFrame({"r": fwd, "ok": same_day.to_numpy()}, index=h1.index)
        f = f[f.ok & f.r.notna()]
        f["dow"] = f.index.dayofweek; f["hr"] = f.index.hour; f["yr"] = f.index.year
        for (dw, hr), g in f.groupby(["dow", "hr"]):
            if len(g) < 150:
                continue
            m = g.r.mean(); t = m / g.r.std() * np.sqrt(len(g))
            ym = g.groupby("yr").r.mean()
            h1m = g[g.yr <= 2022].r.mean(); h2m = g[g.yr >= 2023].r.mean()
            rows.append({"dow": dw, "hour": hr, "hold_h": hold, "n": len(g), "bps": round(m, 2), "t": round(t, 2),
                         "yrs_same": round((np.sign(ym) == np.sign(m)).mean(), 2), "h1_bps": round(h1m, 2),
                         "h2_bps": round(h2m, 2), "halves_agree": bool(np.sign(h1m) == np.sign(h2m))})
    T = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
    (lab.REPORTS / "EXP-028").mkdir(exist_ok=True)
    T.to_csv(lab.REPORTS / "EXP-028" / "season_map.csv", index=False)
    print(f"tests: {len(T)}; |t|>3.5: {(T.t.abs() > 3.5).sum()}; |t|>2.5 & halves agree & yrs_same>=0.75: "
          f"{((T.t.abs() > 2.5) & T.halves_agree & (T.yrs_same >= 0.75)).sum()}")
    print(T.head(30).to_string(index=False))
    # weekday totals (whole server day) M1-era and D1 2008-2018
    d1 = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
    d1["r"] = np.log(d1.close / d1.open) * 1e4
    for a, b in (("2008", "2018"), ("2019", "2026")):
        x = d1.loc[a:b]
        g = x.groupby(x.index.dayofweek).r
        print(f"\nweekday open->close bps {a}-{b}:", g.mean().round(1).to_dict(), " t:",
              (g.mean() / g.std() * np.sqrt(g.count())).round(2).to_dict())

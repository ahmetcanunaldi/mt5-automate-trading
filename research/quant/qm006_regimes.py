"""QM-006: latent regime / drift filtering on daily data (DEV: up to 2024-12-31), yearly walk-forward refits.
Models: Gaussian HMM with 2 / 3 states on [r_t, log range_t] (forward filter only), Kalman local-level drift.
Positions for day t+1 decided at the close of t: long/short = sign(E[r_{t+1}|F_t]), long/flat = 1{E > 0}.
Benchmarks: always long, TSMOM (sign of 60-day return). Daily close-to-close returns, cost = round-trip bps x
|change in position|. Reported: Sharpe, excess over always-long (paired t on daily differences), deflated Sharpe."""
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import filters as F, validate as VAL  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

warnings.filterwarnings("ignore")
FILES = {"XAUUSD": ("XAUUSD_D1_2007", 1.5), "NAS100": ("NAS100_D1_long", 1.3), "DJ30": ("DJ30_D1_long", 1.2),
         "GER40": ("GER40_D1_long", 1.3), "EURUSD": ("EURUSD_D1_long", 2.2), "USDJPY": ("USDJPY_D1_long", 2.2)}
OUT = lab.REPORTS / "quant" / "QM-006"


def load(f):
    d = pd.read_parquet(lab.DATA / f"{f}.parquet")[["open", "high", "low", "close"]]
    d = d[(d.index.dayofweek < 5) & (d.high > d.low)].loc[:"2024-12-31"]
    r = np.log(d.close).diff()
    lr = np.log(np.log(d.high / d.low))
    return pd.DataFrame({"r": r, "lr": lr}).dropna()


def positions(df, first_test):
    P = {k: pd.Series(np.nan, index=df.index) for k in ("hmm2_ls", "hmm2_lf", "hmm3_ls", "hmm3_lf", "kal_ls", "kal_lf")}
    X = df[["r", "lr"]].to_numpy()
    for Y in range(first_test, 2025):
        tr = (df.index < f"{Y}-01-01"); te = (df.index >= f"{Y}-01-01") & (df.index < f"{Y + 1}-01-01")
        if te.sum() == 0:
            continue
        for k in (2, 3):
            o = F.hmm_fit_filter(X[tr], X[: te.nonzero()[0][-1] + 1], k)
            mu = o["mu_next"][te[: len(o["mu_next"])]]
            P[f"hmm{k}_ls"][te] = np.sign(mu); P[f"hmm{k}_lf"][te] = (mu > 0).astype(float)
        kd, _ = F.kalman_drift(df.r[tr].to_numpy(), df.r[: te.nonzero()[0][-1] + 1].to_numpy())
        mu = kd.mu.to_numpy()[te[: len(kd)]]
        P["kal_ls"][te] = np.sign(mu); P["kal_lf"][te] = (mu > 0).astype(float)
    P["always_long"] = pd.Series(1.0, index=df.index)
    P["tsmom60"] = np.sign(np.log(np.exp(df.r.cumsum())).diff(60))
    return pd.DataFrame(P)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    rows = []
    for sym, (f, cost) in FILES.items():
        df = load(f)
        first = max(df.index.year.min() + 5, 2012)
        P = positions(df, first)
        P = P[P.index >= f"{first}-01-01"].dropna()
        rn = df.r.shift(-1).reindex(P.index)
        base = (P.always_long * rn).dropna()
        for k in P.columns:
            w = P[k]
            ret = (w * rn - cost * 1e-4 * w.diff().abs().fillna(0)).dropna()
            diff = (ret - base.reindex(ret.index)).dropna()
            row = {"sym": sym, "model": k, "years": f"{first}-2024", "SR": round(VAL.sharpe(ret), 2),
                   "ann_ret%": round(ret.mean() * 252 * 100, 1), "long%": round((w > 0).mean(), 2),
                   "turnover/yr": round(w.diff().abs().sum() / (len(w) / 252), 1),
                   "excess_vs_long_t": round(diff.mean() / diff.std() * np.sqrt(len(diff)), 2) if k != "always_long" else None,
                   "DSR(12 trials)": round(VAL.deflated_sharpe(ret.to_numpy(), 12), 3)}
            rows.append(row)
            led.log("QM-006", "regime_filters", {"sym": sym, "model": k}, {"SR": row["SR"]})
        print(sym, "done", flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))

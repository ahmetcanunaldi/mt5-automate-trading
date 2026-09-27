"""QM-004: daily mean reversion of equity indices as a heteroskedastic AR(1) / discretized OU model.

Model (per index, daily, DEV only):  y_{t+1} = a + b * z_t + e,   z_t = r_t / sigma_t
  r_t      = today's close-to-close log return, sigma_t = HAR forecast of today's volatility made yesterday
  y_{t+1}  = tomorrow's open->close return / sigma_{t+1}   (executable: enter 01:05, exit 23:30)
Variants: z from close-to-close, from open->close, and the cross-index common factor (mean z over NAS/DJ/SP/GER).
Coefficients re-estimated every year on all earlier years (walk-forward, D1 2014-2024).
Trading rule: position = sign(yhat) when |yhat| > k (k in {0, 0.05, 0.1}) -> long / short next day.
Validation: OOS mean return per trade vs always-long, t, placebo (z shuffled within years), SPA over variants."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.features import atr  # noqa: E402
from research.quant import validate as VAL  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

IDX = {"NAS100": "NAS100_D1_long", "DJ30": "DJ30_D1_long", "SP500": None, "GER40": "GER40_D1_long"}
COST = {"NAS100": 1.3, "DJ30": 1.2, "GER40": 1.3}
OUT = lab.REPORTS / "quant" / "QM-004"


def daily(sym):
    d = pd.read_parquet(lab.DATA / f"{IDX[sym]}.parquet")[["open", "high", "low", "close"]]
    d = d[(d.index.dayofweek < 5) & (d.high > d.low)].loc[:"2024-12-31"]
    lr = np.log(d.close).diff()
    # volatility proxy available on D1 for 2013+: EWMA of squared close-to-close returns (HAR needs intraday RV;
    # RV exists only from 2018 -> use the D1 range-based Parkinson variance in a HAR on D1)
    park = (np.log(d.high / d.low) ** 2) / (4 * np.log(2))
    lp = np.log(park.clip(lower=park[park > 0].min()))
    har = pd.DataFrame({"d": lp, "w": lp.rolling(5).mean(), "m": lp.rolling(22).mean()})
    return d, lr, park, har


def har_sigma_wf(park, har, first, last):
    """walk-forward HAR on log Parkinson variance, one-day-ahead sigma known at the close of t for day t+1."""
    y = np.log(park.clip(lower=park[park > 0].min())).shift(-1)
    sig = pd.Series(np.nan, index=park.index)
    for Y in range(first, last + 1):
        tr = (park.index < f"{Y}-01-01"); te = (park.index >= f"{Y}-01-01") & (park.index < f"{Y + 1}-01-01")
        m = tr & har.notna().all(axis=1) & y.notna()
        A = np.column_stack([np.ones(m.sum()), har[m]])
        b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        s2 = (y[m] - A @ b).var()
        At = np.column_stack([np.ones(te.sum()), har[te].fillna(method=None) if False else har[te].ffill()])
        sig[te] = np.sqrt(np.exp(At @ b + s2 / 2))                  # sigma forecast for day t+1, stamped at t
    return sig


def build(sym):
    d, lr, park, har = daily(sym)
    sig_next = har_sigma_wf(park, har, 2015, 2024)                  # forecast for t+1 made at t
    sig_today = sig_next.shift(1)                                    # forecast for t made at t-1
    oc_next = np.log(d.close / d.open).shift(-1)
    return pd.DataFrame({"z_cc": lr / sig_today, "z_oc": np.log(d.close / d.open) / sig_today,
                         "y": oc_next / sig_next, "ret_next_bps": oc_next * 1e4, "sig_next": sig_next}).dropna()


def wf(df, zcol, k):
    out = []
    for Y in range(2016, 2025):
        tr = df.index < f"{Y}-01-01"; te = (df.index >= f"{Y}-01-01") & (df.index < f"{Y + 1}-01-01")
        A = np.column_stack([np.ones(tr.sum()), df.loc[tr, zcol]])
        b, *_ = np.linalg.lstsq(A, df.loc[tr, "y"], rcond=None)
        yhat = b[0] + b[1] * df.loc[te, zcol]
        pos = np.where(yhat > k, 1, np.where(yhat < -k, -1, 0))
        out.append(pd.DataFrame({"pos": pos, "ret": df.loc[te, "ret_next_bps"], "b1": b[1]}, index=df.index[te]))
    return pd.concat(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    B = {s: build(s) for s in COST}
    common = pd.concat({s: B[s].z_cc for s in B}, axis=1).mean(axis=1)
    rows, D = [], {}
    for s, df in B.items():
        df = df.assign(z_common=common.reindex(df.index))
        for zcol in ("z_cc", "z_oc", "z_common"):
            for k in (0.0, 0.05, 0.1):
                o = wf(df, zcol, k)
                tr = o[o.pos != 0]
                pnl = tr.pos * tr.ret - COST[s]
                long_all = o.ret - COST[s]
                stat = lambda z: float(wf(df.assign(**{zcol: z}), zcol, k).pipe(lambda q: (q.pos * q.ret - COST[s] * (q.pos != 0)).mean()))  # noqa: E731
                row = {"sym": s, "z": zcol, "k": k, "trades": len(tr), "long%": round((tr.pos > 0).mean(), 2),
                       "bps/trade": round(pnl.mean(), 2), "t": round(pnl.mean() / pnl.std() * np.sqrt(len(pnl)), 2),
                       "short_bps": round((tr[tr.pos < 0].pos * tr[tr.pos < 0].ret - COST[s]).mean(), 2) if (tr.pos < 0).any() else None,
                       "always_long_bps": round(long_all.mean(), 2), "b1_last": round(o.b1.iloc[-1], 3),
                       "b1_range": f"{o.b1.min():.3f}..{o.b1.max():.3f}"}
                if zcol == "z_cc" and k == 0.0:
                    pl = VAL.placebo(stat, df[zcol], n=100)
                    row.update({"placebo_p": pl["p"], "placebo_p95": round(pl["null_p95"], 2)})
                rows.append(row); print(row, flush=True)
                led.log("QM-004", "index_daily_ar", {"sym": s, "z": zcol, "k": k}, {"bps": row["bps/trade"], "t": row["t"]})
                D[f"{s}|{zcol}|{k}"] = (o.pos * o.ret - COST[s] * (o.pos != 0)).rename(f"{s}|{zcol}|{k}")
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    M = pd.concat(D.values(), axis=1).fillna(0.0)
    spa = VAL.spa_test(M.to_numpy(), B=500)
    pd.set_option("display.width", 230)
    print(T.to_string(index=False))
    print("SPA over all", M.shape[1], "variants (benchmark = no trade):", spa)

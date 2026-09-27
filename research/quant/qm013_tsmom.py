"""QM-013: diversified volatility-scaled time-series momentum (Moskowitz, Ooi & Pedersen 2012) on 8 assets,
D1 2013-2024 (DEV), weekly rebalancing (decision Friday close, positions from Monday open, weekend flat =
Monday's return counted from the open).
Per asset: signal s = sign(return over L days) (L = 20/60/120/250 and the average of the four), size
w = s * sigma_target / sigma_hat (EWMA 60-day, known at the decision), capped at 2x.
Portfolio weights across assets: equal risk (1/N), HRP and Ledoit-Wolf min-variance-tilted on trailing 2 years of
the per-asset strategy returns (walk-forward, refit yearly). Benchmark: long-only risk parity (same sizing, s = +1).
Costs: turnover x round-trip bps. Reported: Sharpe, excess vs benchmark (paired t), DSR over the variants."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import portfolio as P, validate as VAL  # noqa: E402
from research.quant.qm001_map import COST  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

FILES = {"XAUUSD": "XAUUSD_D1_long", "XAGUSD": "XAGUSD_D1_long", "NAS100": "NAS100_D1_long", "DJ30": "DJ30_D1_long",
         "SP500": None, "GER40": "GER40_D1_long", "EURUSD": "EURUSD_D1_long", "USDJPY": "USDJPY_D1_long"}
OUT = lab.REPORTS / "quant" / "QM-013"


def load():
    C, O = {}, {}
    for s, f in FILES.items():
        if f is None:
            continue
        d = pd.read_parquet(lab.DATA / f"{f}.parquet")[["open", "high", "low", "close"]]
        d = d[d.index.dayofweek < 5].loc["2011":"2024-12-31"]
        C[s], O[s] = d.close, d.open
    Cd = pd.DataFrame(C).dropna(); return Cd, pd.DataFrame(O).reindex(Cd.index)


def asset_returns(C, O, L, long_only=False):
    r = np.log(C).diff()
    mon = C.index.dayofweek == 0
    r.loc[mon] = np.log(C / O).loc[mon]                                 # weekend flat: Monday from the open
    sig_hat = r.ewm(span=60).std()
    fri = C.index.dayofweek == 4
    if long_only:
        s = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    elif L == "avg":
        s = sum(np.sign(np.log(C).diff(k)) for k in (20, 60, 120, 250)) / 4
    else:
        s = np.sign(np.log(C).diff(L))
    w = (s * (0.10 / np.sqrt(252)) / sig_hat).clip(-2, 2)
    w = w.where(pd.Series(fri, index=C.index), axis=0).ffill().shift(1)   # decided Friday close, held next week
    cost = pd.DataFrame({c: COST.get(c, 1.5) for c in C.columns}, index=C.index) * 1e-4
    ret = w * r - w.diff().abs() * cost
    return ret.dropna()


def combine(R, method):
    out = pd.Series(0.0, index=R.index); W = {}
    for Y in range(2015, 2025):
        tr = R.loc[f"{Y - 2}":f"{Y - 1}"]; te = R.index.year == Y
        if method == "equal":
            w = np.ones(R.shape[1]) / R.shape[1]
        elif method == "hrp":
            w = P.hrp_weights(P.ledoit_wolf_cov(tr.to_numpy()))
        else:                                                           # min-variance tilt (LW), long-only weights
            cov = P.ledoit_wolf_cov(tr.to_numpy()); iv = np.linalg.pinv(cov) @ np.ones(len(cov))
            w = np.clip(iv, 0, None); w = w / w.sum() if w.sum() > 0 else np.ones(len(cov)) / len(cov)
        W[Y] = w
        out[te] = R[te].to_numpy() @ w * R.shape[1]                     # scale so weights sum to N (comparable risk)
    return out.loc["2015":]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    C, O = load()
    bench = combine(asset_returns(C, O, None, long_only=True), "equal")
    rows = []
    for L in (20, 60, 120, 250, "avg"):
        R = asset_returns(C, O, L)
        per = {c: round(VAL.sharpe(R[c].loc["2015":]), 2) for c in R.columns}
        for method in ("equal", "hrp", "lw_minvar"):
            p = combine(R, method)
            diff = (p - bench.reindex(p.index)).dropna()
            row = {"lookback": L, "weights": method, "SR": round(VAL.sharpe(p), 2), "ann_ret%": round(p.mean() * 252 * 100, 1),
                   "ann_vol%": round(p.std() * np.sqrt(252) * 100, 1), "SR_bench_longRP": round(VAL.sharpe(bench), 2),
                   "excess_t": round(diff.mean() / diff.std() * np.sqrt(len(diff)), 2),
                   "corr_with_bench": round(p.corr(bench.reindex(p.index)), 2),
                   "DSR(15)": round(VAL.deflated_sharpe(p.to_numpy(), 15), 3), "per_asset_SR": per if method == "equal" else ""}
            rows.append(row); print({k: v for k, v in row.items() if k != "per_asset_SR"}, flush=True)
            led.log("QM-013", "tsmom_portfolio", {"L": L, "w": method}, {"SR": row["SR"]})
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 150)
    print(T.to_string(index=False))
    yearly = combine(asset_returns(C, O, "avg"), "equal").groupby(lambda t: t.year).apply(VAL.sharpe)
    print("TSMOM(avg, equal) Sharpe by year:", yearly.round(2).to_dict())

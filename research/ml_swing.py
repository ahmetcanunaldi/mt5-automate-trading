"""EXP-041: swing-horizon ML on daily bars 2008-2026 (gold only features), yearly walk-forward from 2012.
Label: sign of the 5-day forward return (close t+5 / open t+1). Trade: at the next open in the predicted direction
when |p-0.5| > delta, hold 5 days, SL 2 ATR_D; executed with the swing engine (swaps included)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research import engine, lab, metrics  # noqa: E402
from research.features import atr  # noqa: E402
from research.run_swing import COSTS, exec_d1, guards  # noqa: E402

if __name__ == "__main__":
    d = pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet")
    c = d.close; a = atr(d, 20)
    F = pd.DataFrame(index=d.index)
    for n in (1, 3, 5, 10, 20, 60, 120, 250):
        F[f"r{n}"] = np.log(c / c.shift(n)) / (a / c * np.sqrt(n))
    for n in (20, 60, 250):
        F[f"hi{n}"] = (c - d.high.rolling(n).max()) / a
        F[f"lo{n}"] = (c - d.low.rolling(n).min()) / a
    F["vol_ratio"] = a / atr(d, 120)
    F["dow"] = d.index.dayofweek; F["dom"] = d.index.day
    F["rng1"] = (d.high - d.low) / a
    F["clv"] = ((c - d.low) - (d.high - c)) / (d.high - d.low).replace(0, np.nan)
    fwd = np.log(c.shift(-5) / d.open.shift(-1))
    y = (fwd > 0).astype(int)
    ok = fwd.notna() & F.notna().all(axis=1)
    p = pd.Series(np.nan, index=d.index)
    for yr in range(2012, 2027):
        tr = ok & (d.index.year < yr) & (d.index < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=10))
        te = ok.index.year == yr
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=60,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1)
        mdl.fit(F[tr], y[tr])
        p[te & F.notna().all(axis=1)] = mdl.predict_proba(F[te & F.notna().all(axis=1)])[:, 1]
    m = p.notna() & fwd.notna()
    ym, pm = y[m], p[m]
    print("AUC 2012-2026:", round(roc_auc_score(ym, pm), 3),
          " by year:", {int(k): round(roc_auc_score(ym[ym.index.year == k], pm[pm.index.year == k]), 3)
                        for k in range(2012, 2027) if (ym.index.year == k).sum() > 50})
    v = np.sign(p[m] - 0.5) * fwd[m] * 1e4
    print("signed 5d fwd bps:", round(v.mean(), 1), " by year:", v.groupby(v.index.year).mean().round(0).to_dict())
    _, x = exec_d1()
    for delta in (0.0, 0.05, 0.1):
        sel = m & ((p - 0.5).abs() > delta)
        dirn = np.sign(p[sel] - 0.5).astype(int)
        s = pd.DataFrame({"dir": dirn.to_numpy(), "sl": (2.0 * a[sel]).to_numpy(), "tp": (50 * a[sel]).to_numpy(),
                          "hold_min": 5 * 1440, "be": 0.0, "trail": 0.0, "leg": "ml_swing"},
                         index=sel[sel].index + pd.Timedelta(days=1))
        res = engine.run(x, s.loc["2012":], guards(False), COSTS)
        t = res.trades
        mm = metrics.summarize(res, f"ml_swing d{delta}")
        yr = t.groupby(t.entry_time.dt.year).R.sum()
        print(f"delta {delta}: n {len(t)} avgR {mm['avg_R']} SR {mm['sharpe']} net {mm['net_pct']}% DD {mm['max_total_dd_pct']}% "
              f"yrs+ {(yr > 0).sum()}/{len(yr)}  {yr.round(1).to_dict()}")

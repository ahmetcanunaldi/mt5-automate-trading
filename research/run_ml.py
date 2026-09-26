"""EXP-006: LightGBM meta-labeling on broad breakout candidates (DEV walk-forward -> VAL)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research import calendar_news, features, lab, ml_meta  # noqa: E402
from research.strategies.rules import FAMILIES  # noqa: E402

EXITS = {"rr": 3.0, "trail_atr": 0.0, "be_r": 0.0, "hold_min": 480}
PRIMARY = [
    ("donchian", {"n": 32, "sl_atr": 2.0, "start": 120, "end": 1320, "trend": "none", "per_day": 3, "min_width_atr": 0.0}),
    ("ny_orb", {"or_start": 990, "or_bars": 2, "end": 1200, "sl_atr": 2.0, "trend": "none", "min_rng_atr": 0.3, "max_rng_atr": 8.0}),
    ("prev_day_break", {"start": 120, "end": 1320, "sl_atr": 2.0, "trend": "none", "buffer_atr": 0.1}),
    ("asia_breakout", {"start": 600, "end": 900, "sl_atr": 2.0, "min_rng_atr": 0.5, "max_rng_atr": 10.0, "trend": "none"}),
]


def candidates(df):
    parts = []
    for k, (fam, p) in enumerate(PRIMARY):
        s = FAMILIES[fam](df, **p, **EXITS)
        s["src"] = k
        parts.append(s)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]


if __name__ == "__main__":
    df = features.base_frame(lab.signal_bars("M15"), lab.signal_bars("H1"))
    news = calendar_news.load_news_server_times()
    F = ml_meta.feature_frame(df, news)
    F.index = df["close_time"].values
    sig = candidates(df)
    lbl = ml_meta.triple_barrier(df, sig)
    X = F.reindex(sig.index).copy()
    X["dir"] = sig["dir"].values
    X["src"] = sig["src"].values
    # direction-aware features: multiply momentum-like features by dir so the model sees "with/against"
    for col in ("ret1", "ret4", "ret16", "ret64", "vwap_dist", "h1_trend", "h1_pos", "body", "clv"):
        X[col + "_d"] = X[col] * X["dir"]
    X["sl_atr"] = sig["sl"].values / df.set_index("close_time")["atr"].reindex(sig.index).values
    X["tp_atr"] = sig["tp"].values / df.set_index("close_time")["atr"].reindex(sig.index).values
    data = X.join(lbl).dropna(subset=["y"])
    cols = [c for c in data.columns if c not in ("y", "R")]
    dev = data.loc[: lab.PERIODS["DEV"][1]]
    val = data.loc[lab.PERIODS["VAL"][0]: lab.PERIODS["VAL"][1]]
    print("candidates DEV/VAL:", len(dev), len(val), "base win DEV", round(dev.y.mean(), 3), "mean R", round(dev.R.mean(), 3))

    oof = pd.Series(np.nan, index=dev.index)
    for tr, te in ml_meta.purged_folds(dev.index, n_folds=4):
        mdl = ml_meta.lgbm().fit(dev.loc[tr, cols], dev.loc[tr, "y"])
        oof[te] = mdl.predict_proba(dev.loc[te, cols])[:, 1]
    m = oof.notna()
    print("OOF AUC", round(roc_auc_score(dev.y[m], oof[m]), 3))
    q = pd.qcut(oof[m], 5, labels=False)
    print("OOF quintile -> mean R:", dev.R[m].groupby(q).mean().round(3).to_dict(), "count", q.value_counts().sort_index().to_dict())

    mdl = ml_meta.lgbm().fit(dev[cols], dev["y"])
    pv = pd.Series(mdl.predict_proba(val[cols])[:, 1], index=val.index)
    print("VAL AUC", round(roc_auc_score(val.y, pv), 3))
    qv = pd.qcut(pv, 5, labels=False)
    print("VAL quintile -> mean R:", val.R.groupby(qv).mean().round(3).to_dict())
    imp = pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False)
    print("top features:", imp.head(12).to_dict())

    results = {}
    for thr_q in (0.5, 0.6, 0.7, 0.8):
        thr = float(np.quantile(oof[m], thr_q))
        s_dev = sig.loc[oof[m][oof[m] >= thr].index]
        s_val = sig.loc[pv[pv >= thr].index]
        for per, s in (("DEVoof", s_dev), ("VAL", s_val)):
            if per == "DEVoof":
                s = s.copy()
                mm, _ = lab.evaluate(s, "DEV", label=f"ml_thr{thr_q}")
            else:
                mm, _ = lab.evaluate(s, "VAL", label=f"ml_thr{thr_q}")
            results[f"thr{thr_q}|{per}"] = mm
            print(per, lab.fmt(mm), flush=True)
    lab.save_experiment("EXP-006", {"primary": PRIMARY, "exits": EXITS, "features": cols}, results)

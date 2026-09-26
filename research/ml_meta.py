"""ML layer: triple-barrier labels + LightGBM meta-labeling / direction models.

Candidates come from a primary rule (signals frame). For each candidate we compute features known at the
decision time and a label from the M15 path: +1 if TP is touched before SL within the holding horizon
(pessimistic: SL first when both inside one bar), else 0. A LightGBM classifier is trained with purged,
embargoed walk-forward folds on DEV; the out-of-fold probabilities pick a threshold; VAL is scored with the
model refit on all of DEV.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "hour", "dow", "atr_pct", "atr_ratio", "ret1", "ret4", "ret16", "ret64", "vwap_dist", "dh_dist", "dl_dist",
    "h1_trend", "h1_pos", "body", "clv", "vol_ratio", "spread", "news_prev_h", "news_next_h", "dir",
    "rng_day", "sl_atr", "tp_atr",
]


def feature_frame(df: pd.DataFrame, news_times: pd.DatetimeIndex) -> pd.DataFrame:
    a = df["atr"]
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["hour"] = df.index.hour + df.index.minute / 60
    f["dow"] = df["dow"]
    f["atr_pct"] = a / c * 1e4
    f["atr_ratio"] = a / df["atr_slow"]
    for k in (1, 4, 16, 64):
        f[f"ret{k}"] = (c - c.shift(k)) / a
    f["vwap_dist"] = (c - df["vwap"]) / a
    f["dh_dist"] = (c - df["dh"]) / a
    f["dl_dist"] = (c - df["dl"]) / a
    f["h1_trend"] = (df["h1_ema50"] - df["h1_ema200"]) / df["h1_atr"]
    f["h1_pos"] = (df["h1_close"] - df["h1_ema50"]) / df["h1_atr"]
    f["body"] = (c - df["open"]) / a
    f["clv"] = ((c - df["low"]) - (df["high"] - c)) / (df["high"] - df["low"]).replace(0, np.nan)
    f["vol_ratio"] = df["tick_volume"] / df["tick_volume"].rolling(96).mean()
    f["spread"] = df["spread"]
    day_hi = df.groupby("date")["high"].cummax()
    day_lo = df.groupby("date")["low"].cummin()
    f["rng_day"] = (day_hi - day_lo) / a
    t = df["close_time"].to_numpy().astype("datetime64[ns]")
    n = np.sort(news_times.values.astype("datetime64[ns]"))
    i = np.searchsorted(n, t)
    nxt = np.where(i < len(n), n[np.minimum(i, len(n) - 1)], t + np.timedelta64(30, "D"))
    prv = np.where(i > 0, n[np.maximum(i - 1, 0)], t - np.timedelta64(30, "D"))
    f["news_next_h"] = np.minimum((nxt - t) / np.timedelta64(1, "h"), 72)
    f["news_prev_h"] = np.minimum((t - prv) / np.timedelta64(1, "h"), 72)
    return f


def triple_barrier(df: pd.DataFrame, sig: pd.DataFrame, bar_minutes=15) -> pd.Series:
    """Label each signal (indexed by decision time = bar close) on the following bars' path."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = np.maximum(df["spread"].to_numpy(), 15) * 0.01
    close_t = df["close_time"].to_numpy().astype("datetime64[ns]")
    day = df.index.values.astype("datetime64[D]")
    pos = np.searchsorted(close_t, sig.index.values.astype("datetime64[ns]"))
    lab = np.full(len(sig), np.nan)
    rr = np.full(len(sig), np.nan)
    for k, (p, d, sl, tp, hold) in enumerate(zip(pos, sig["dir"], sig["sl"], sig["tp"], sig["hold_min"])):
        if p >= len(c) - 1:
            continue
        e = p + 1                      # entry at next bar open
        ep = o[e] + (spr[e] if d == 1 else 0.0)
        nb = int(np.ceil(hold / bar_minutes))
        res = None
        for j in range(e, min(e + nb, len(c))):
            if day[j] != day[e]:
                break
            if d == 1:
                if l[j] <= ep - sl:
                    res = -1.0; break
                if h[j] >= ep + tp:
                    res = tp / sl; break
            else:
                if h[j] + spr[j] >= ep + sl:
                    res = -1.0; break
                if l[j] + spr[j] <= ep - tp:
                    res = tp / sl; break
            last = j
        if res is None:
            j = min(e + nb, len(c)) - 1
            xp = c[j] if d == 1 else c[j] + spr[j]
            res = (xp - ep) * d / sl
        rr[k] = res
        lab[k] = 1.0 if res > 0 else 0.0
    return pd.DataFrame({"y": lab, "R": rr}, index=sig.index)


def purged_folds(times: pd.DatetimeIndex, n_folds=4, embargo="3D", min_train_frac=0.3):
    """Expanding-window walk-forward folds with an embargo gap between train and test."""
    t = pd.Series(times)
    start = t.iloc[int(len(t) * min_train_frac)]
    edges = pd.date_range(start, t.iloc[-1], periods=n_folds + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = (t < a - pd.Timedelta(embargo)).to_numpy()
        te = ((t >= a) & (t < b)).to_numpy()
        yield tr, te


def lgbm(seed=0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              random_state=seed, verbose=-1)

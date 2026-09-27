"""FXR-003: pooled multi-signal model ("combine many weak signals") on the 7-currency panel.

Rows = (day t at server hour 9, currency c incl. USD). Target = next-day cross-sectionally demeaned currency return
divided by its 60-day vol. Features (all known at t, cross-sectionally demeaned where meaningful):
  FXR-001 price signals (momentum 1..250, 12-1, MA gaps, PCA residuals, low vol, last 4h/8h), carry level and
  3-month carry change (FRED, 2-month lag), equity risk-on exposure (rolling 120-day beta of the currency to DJ30
  and GER40 x their last 1-day / 5-day return), gold exposure (beta to XAUUSD x gold 1d/5d), day-of-week,
  month-end flag, currency one-hot.
Walk-forward: expanding window, retrain every year, test years 2018..2024 (DEV only). Models: ridge, LightGBM.
Portfolio: XS-demeaned prediction -> weights (pair gross 1), optionally smoothed (EMA) to cut turnover; costs as FXR-001.
Placebo: targets shuffled across days within each year (20 runs) -> null distribution of OOS IC and net SR."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from research import lab  # noqa: E402
from research.fx import data as F  # noqa: E402
from research.fx.fxr001_factors import daily_grid, signals, xs_demean  # noqa: E402
from research.fx.fxr002_flows import costs_table, rates, shp  # noqa: E402

OUT = lab.REPORTS / "fx" / "FXR-003"
rng = np.random.default_rng(3)


def build():
    at, fwd, dret = daily_grid(9)
    Hr = F.ccy_log("H1", end=F.DEV_END)
    S = signals(at, dret, Hr)
    feats = {k: xs_demean(v.reindex(at.index)) for k, v in S.items() if k != "vol_change"}
    R = rates().reindex(at.index, method="ffill")[at.columns]
    feats["carry"] = xs_demean(R)
    feats["carry_chg"] = xs_demean(R - R.shift(63))
    ext = {"DJ30": pd.read_parquet(lab.DATA / "DJ30_D1_long.parquet").close,
           "GER40": pd.read_parquet(lab.DATA / "GER40_D1_long.parquet").close,
           "XAU": pd.read_parquet(lab.DATA / "XAUUSD_D1_2007.parquet").close}
    for k, px in ext.items():
        lr = np.log(px).diff()
        lr.index = lr.index.normalize()
        # returns up to the previous server day (D1 bar of day t-1 is complete at t 09:00)
        r1 = lr.shift(1).reindex(at.index.normalize()).set_axis(at.index)
        r5 = lr.rolling(5).sum().shift(1).reindex(at.index.normalize()).set_axis(at.index)
        beta = pd.DataFrame({c: dret[c].rolling(120).cov(r1) / r1.rolling(120).var() for c in at.columns})
        feats[f"{k}_x1"] = xs_demean(beta.mul(r1, axis=0))
        feats[f"{k}_x5"] = xs_demean(beta.mul(r5, axis=0))
    vol = dret.iloc[:, 1:].rolling(60).std(); vol.insert(0, "USD", vol.mean(1))
    Y = xs_demean(fwd) / vol.reindex(fwd.index)
    rows = []
    for c in at.columns:
        d = pd.DataFrame({k: v[c] for k, v in feats.items()})
        d["ccy"] = c; d["y"] = Y[c].reindex(d.index); d["fwd"] = fwd[c].reindex(d.index)
        rows.append(d)
    P = pd.concat(rows).sort_index()
    P["dow"] = P.index.dayofweek
    nxt = pd.Series(P.index.normalize()).groupby(P.index.to_period("M")).transform("max").to_numpy()
    P["month_end"] = (P.index.normalize() == nxt).astype(float)
    P = pd.get_dummies(P, columns=["ccy"], dtype=float)
    P["ccy"] = pd.concat(rows).sort_index()["ccy"]
    P.index.name = "time"
    return P.dropna(subset=["y"]), at


def walk_forward(P, model, y_col="y", years=range(2018, 2025)):
    X_cols = [c for c in P.columns if c not in ("y", "fwd", "ccy")]
    pred = pd.Series(np.nan, index=range(len(P)))
    P = P.reset_index()
    for yv in years:
        tr = (P["time"].dt.year < yv) & (P["time"].dt.year >= 2015)
        te = P["time"].dt.year == yv
        Xtr = P.loc[tr, X_cols].fillna(0); Xte = P.loc[te, X_cols].fillna(0)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        m = model()
        m.fit((Xtr - mu) / sd, P.loc[tr, y_col].clip(-5, 5))
        pred[te[te].index] = m.predict((Xte - mu) / sd)
    P["pred"] = pred.to_numpy()
    return P[P.pred.notna()]


def portfolio(Pp, costs, ema=1):
    Wp = Pp.pivot_table(index="time", columns="ccy", values="pred")
    Wp = Wp.sub(Wp.mean(1), axis=0)
    if ema > 1:
        Wp = Wp.ewm(span=ema).mean()
    W = Wp.drop(columns="USD")
    W = W.div(W.abs().sum(1), axis=0)
    fw = Pp.pivot_table(index="time", columns="ccy", values="fwd").drop(columns="USD")
    gross = (W * fw).sum(1) * 1e4
    turn = W.diff().abs().fillna(0)
    cost = (turn * costs.reindex(W.index.year).set_axis(W.index)[W.columns]).sum(1)
    return gross, gross - cost


def ic(Pp):
    return np.array([stats.spearmanr(g.pred, g.y)[0] for _, g in Pp.groupby("time") if g.y.notna().sum() > 3])


MODELS = {"ridge": lambda: Ridge(alpha=1000.0),
          "lgbm": lambda: LGBMRegressor(n_estimators=200, learning_rate=0.02, num_leaves=8, min_child_samples=200,
                                        subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1)}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    P, at = build()
    costs = costs_table()
    print("rows", len(P), "features", len([c for c in P.columns if c not in ("y", "fwd", "ccy")]))
    res = {}
    for name, mk in MODELS.items():
        Pp = walk_forward(P, mk)
        i = ic(Pp)
        out = {"ic": round(i.mean(), 4), "ic_t": round(i.mean() / i.std() * np.sqrt(len(i)), 2)}
        for ema in (1, 5, 20):
            g, n = portfolio(Pp, costs, ema)
            out[f"SRg_ema{ema}"] = shp(g); out[f"SRn_ema{ema}"] = shp(n)
        g, n = portfolio(Pp, costs, 5)
        out["by_year_net_ema5"] = n.groupby(n.index.year).apply(shp).to_dict()
        # placebo: shuffle targets across days within each year
        null_ic, null_sr = [], []
        for _ in range(20 if name == "ridge" else 8):
            Q = P.copy().reset_index()
            for yv, g_ in Q.groupby(Q["time"].dt.year):
                days = g_["time"].unique(); perm = dict(zip(days, rng.permutation(days)))
                src = Q.loc[g_.index].set_index(["time", "ccy"])[["y", "fwd"]]
                key = pd.MultiIndex.from_arrays([g_["time"].map(perm), g_["ccy"]])
                Q.loc[g_.index, ["y", "fwd"]] = src.reindex(key).to_numpy()
            Qp = walk_forward(Q.set_index("time"), mk)
            null_ic.append(ic(Qp).mean()); null_sr.append(shp(portfolio(Qp, costs, 5)[1]))
        out["placebo_p_ic"] = float((np.array(null_ic) >= i.mean()).mean())
        out["placebo_ic_mean"] = round(float(np.mean(null_ic)), 4)
        out["placebo_p_SRn5"] = float((np.array(null_sr) >= out["SRn_ema5"]).mean())
        res[name] = out
        print(name, out, flush=True)
    pd.DataFrame(res).to_csv(OUT / "pooled.csv")

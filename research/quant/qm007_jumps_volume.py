"""QM-007: jumps, self-excitation and volume (DEV 2018-2024).
(a) Lee-Mykland jumps on intraday 5-min returns (alpha 1 %), split news (release within +-15 min) / non-news;
    post-jump returns signed by the jump direction over (+5..+15], (+15..+60], (+60..+240] minutes, t-stats and bps
    vs cost; entries in the news case are only allowed >= 10 min after the release (rule v2), so (+15..) windows
    are the executable ones.
(b) Hawkes (exponential) fit to non-news jump times per year: branching ratio = share of jumps triggered by jumps.
(c) Campbell-Grossman-Wang: r_{t+1} = a + b r_t + c r_t V_t, V_t = log tick volume minus its trailing mean for the
    same time of day (15-min, 1h) or trailing 20-day mean (daily); robust (HC) t of c."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q, jumps as J  # noqa: E402
from research.quant.qm001_map import COST  # noqa: E402
from research import symbols  # noqa: E402

OUT = lab.REPORTS / "quant" / "QM-007"


def post_jump(sym):
    b = Q.bars(sym, "5min")
    lr = np.log(b.close)
    r = lr.diff()
    same = b.index.normalize() == pd.Series(b.index, index=b.index).shift(1).dt.normalize()
    r = r.where(same)
    L, crit = J.lee_mykland(r.fillna(0).to_numpy(), K=78)
    jm = (np.abs(L) > crit) & same.to_numpy() & np.isfinite(L)
    news = symbols.news_times(sym)
    t = b.index[jm]
    near = np.array([np.abs((news - x).total_seconds()).min() <= 900 if len(news) else False for x in t]) if len(t) else np.array([])
    sign = np.sign(r[jm].to_numpy())
    close = b.close
    rows = []
    for lab_, a_, b_ in (("+5..15", 1, 3), ("+15..60", 3, 12), ("+60..240", 12, 48)):
        pa = close.shift(-a_).reindex(t).to_numpy(); pb = close.shift(-b_).reindex(t).to_numpy()
        ok_day = (pd.Series(b.index, index=b.index).shift(-b_).dt.normalize().reindex(t).to_numpy() == t.normalize().to_numpy())
        v = sign * np.log(pb / pa) * 1e4
        for grp, m in (("news", near), ("non-news", ~near)):
            x = v[m & ok_day & np.isfinite(v)]
            if len(x) > 20:
                rows.append({"sym": sym, "jumps": grp, "window": lab_, "n": len(x), "bps_signed": round(x.mean(), 2),
                             "t": round(x.mean() / x.std() * np.sqrt(len(x)), 2), "cost": COST[sym]})
    hk = []
    tn = t[~near] if len(t) else t
    for y in range(2018, 2025):
        ty = tn[tn.year == y]
        if len(ty) > 30:
            s = (ty - pd.Timestamp(f"{y}-01-01")).total_seconds().to_numpy() / 3600.0
            p = J.hawkes_fit(s, (pd.Timestamp(f"{y + 1}-01-01") - pd.Timestamp(f"{y}-01-01")).total_seconds() / 3600)
            hk.append(p["branching"])
    return rows, {"sym": sym, "jumps": int(jm.sum()), "news_share": round(float(near.mean()), 2) if len(near) else 0,
                  "hawkes_branching_mean": round(float(np.mean(hk)), 2) if hk else None, "years": len(hk)}


def cgw(sym, tf):
    b = Q.bars(sym, tf)
    r = np.log(b.close).diff()
    lv = np.log(b.tick_volume.clip(lower=1))
    if tf == "1D":
        V = lv - lv.rolling(20).mean().shift(1)
    else:
        tod = b.index.hour * 60 + b.index.minute
        V = lv - lv.groupby(tod).transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
        same = b.index.normalize() == pd.Series(b.index, index=b.index).shift(1).dt.normalize()
        r = r.where(same)
    df = pd.DataFrame({"y": r.shift(-1), "r": r, "rV": r * V}).dropna()
    df = df.loc[:, :] * [1e4, 1e4, 1e4]
    X = sm.add_constant(df[["r", "rV"]])
    m = sm.OLS(df.y, X).fit(cov_type="HC1")
    return {"sym": sym, "tf": tf, "n": len(df), "b_r": round(m.params["r"], 4), "t_r": round(m.tvalues["r"], 2),
            "c_rV": round(m.params["rV"], 4), "t_rV": round(m.tvalues["rV"], 2)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    PJ, HK, CG = [], [], []
    for sym in Q.SYMS:
        rows, hk = post_jump(sym); PJ += rows; HK.append(hk)
        for tf in ("15min", "1h", "1D"):
            CG.append(cgw(sym, tf))
        print(sym, hk, flush=True)
    pd.set_option("display.width", 220)
    A, H, C = pd.DataFrame(PJ), pd.DataFrame(HK), pd.DataFrame(CG)
    A.to_csv(OUT / "post_jump.csv", index=False); H.to_csv(OUT / "hawkes.csv", index=False); C.to_csv(OUT / "cgw.csv", index=False)
    print(A.to_string(index=False)); print(H.to_string(index=False)); print(C.to_string(index=False))

"""QM-011: volatility surprise and intraday continuation (DEV 2018-2024).
For each day: morning window return and realized variance; HAR forecast of the day's RV (walk-forward) times the
morning's typical share of daily RV (trailing, by symbol) -> expected morning RV. Surprise S = realized / expected.
Test: sign(morning return) x rest-of-day return, overall and by S tercile (thresholds from the training years),
plus the high-S & large-move subset. Windows (server time):
  indices  : morning 16:30-18:00 (US cash open), rest 18:00-23:00;   GER40 also 10:00-12:00 -> 12:00-16:30
  XAU/XAG/FX: morning 01:05-10:00 (Asia), rest 10:00-23:00;  and London 10:00-12:00 -> 12:00-23:00"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q, vol as V  # noqa: E402
from research.quant.qm001_map import COST  # noqa: E402
from research.quant.validate import TrialLedger  # noqa: E402

OUT = lab.REPORTS / "quant" / "QM-011"
WIN = {"NAS100": [((990, 1080), (1080, 1380))], "DJ30": [((990, 1080), (1080, 1380))], "SP500": [((990, 1080), (1080, 1380))],
       "GER40": [((990, 1080), (1080, 1380)), ((600, 720), (720, 990))],
       "XAUUSD": [((65, 600), (600, 1380)), ((600, 720), (720, 1380))], "XAGUSD": [((65, 600), (600, 1380))],
       "EURUSD": [((65, 600), (600, 1380)), ((600, 720), (720, 1380))], "USDJPY": [((65, 600), (600, 1380))]}


def day_windows(sym, w_morn, w_rest):
    b = Q.bars(sym, "5min")
    tm = b.index.hour * 60 + b.index.minute
    r = np.log(b.close).diff()
    same = b.index.normalize() == pd.Series(b.index, index=b.index).shift(1).dt.normalize()
    r = r.where(same)
    day = b.index.normalize()
    mm = (tm >= w_morn[0]) & (tm < w_morn[1]); rr = (tm >= w_rest[0]) & (tm < w_rest[1])
    D = pd.DataFrame({"m_ret": r[mm].groupby(day[mm]).sum(), "m_rv": (r[mm] ** 2).groupby(day[mm]).sum(),
                      "rest": r[rr].groupby(day[rr]).sum()})
    R = Q.realized(sym); R = R[R.rv > 0]
    F = V.walk_forward_rv(R, 2019, 2024)
    share = (D.m_rv / R.rv.reindex(D.index)).rolling(250, min_periods=60).median().shift(1)
    D["S"] = D.m_rv / (F.har_lev_j.reindex(D.index) * share)
    D["sig"] = np.sqrt(F.har_lev_j.reindex(D.index))
    return D.dropna()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    led = TrialLedger()
    rows = []
    for sym, wins in WIN.items():
        for wm, wr in wins:
            D = day_windows(sym, wm, wr)
            D = D.loc["2020":]
            x = np.sign(D.m_ret) * D.rest * 1e4
            q1, q2 = D.S.loc[:"2021"].quantile([1 / 3, 2 / 3])          # thresholds from the first two test years
            te = D.loc["2022":]; xt = x.loc["2022":]
            res = {"sym": sym, "morning": f"{wm[0]//60:02d}:{wm[0]%60:02d}-{wm[1]//60:02d}:{wm[1]%60:02d}", "n": len(x),
                   "all_bps": round(x.mean(), 2), "all_t": round(x.mean() / x.std() * np.sqrt(len(x)), 2)}
            for lab_, m in (("S_low", te.S <= q1), ("S_mid", (te.S > q1) & (te.S <= q2)), ("S_high", te.S > q2)):
                v = xt[m]
                res[f"{lab_}_bps"] = round(v.mean(), 2); res[f"{lab_}_t"] = round(v.mean() / v.std() * np.sqrt(len(v)), 2)
            big = te.m_ret.abs() > 0.5 * te.sig
            v = xt[(te.S > q2) & big]
            res.update({"highS_bigmove_n": len(v), "highS_bigmove_bps": round(v.mean(), 2),
                        "highS_bigmove_t": round(v.mean() / v.std() * np.sqrt(len(v)), 2) if len(v) > 2 else None, "cost": COST[sym]})
            rows.append(res); print(res, flush=True)
            led.log("QM-011", "vol_surprise", {"sym": sym, "win": res["morning"]}, {"t_high": res["S_high_t"]})
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))

"""QM-002: volatility forecasting on DEV 2018-2024 (walk-forward 2020-2024): naive, EWMA, HAR, HAR+leverage+jump,
GARCH, GJR-GARCH; QLIKE and Mincer-Zarnowitz R^2 against next-day realized variance (5-min). Rough-volatility
Hurst exponent of log sqrt(RV)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from research import lab  # noqa: E402
from research.quant import data as Q, vol as V  # noqa: E402

warnings.filterwarnings("ignore")
OUT = lab.REPORTS / "quant" / "QM-002"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in Q.SYMS:
        R = Q.realized(sym)
        R = R[R.rv > 0]
        F = V.walk_forward_rv(R, 2020, 2024)
        F.to_csv(OUT / f"forecasts_{sym}.csv")
        h, _ = V.rough_hurst(R.rv)
        row = {"sym": sym, "days": len(F), "H_rough": round(h, 3)}
        for mdl in ("naive", "ewma", "har", "har_lev_j", "garch", "gjr"):
            row[f"QLIKE_{mdl}"] = round(V.qlike(F.rv, F[mdl]), 4)
        for mdl in ("har", "har_lev_j", "gjr"):
            row[f"R2_{mdl}"] = round(V.mz_r2(F.rv, F[mdl]), 3)
        rows.append(row); print(row, flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))

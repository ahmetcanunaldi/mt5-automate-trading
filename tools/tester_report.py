"""Parse an MT5 Strategy Tester report (saved via MCP tester_get_report as xlsx) -> deals table, balance curve,
per-symbol / per-leg summary and a chart with the FundingPips limits.
usage: python tools/tester_report.py <run_id> <name>   (writes reports/tester/<name>/...)"""
import pathlib
import shutil
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deploy import MQL5_DIR  # noqa: E402
from mcp_client import MT5MCP  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def fetch(run_id, name):
    out = ROOT / "reports" / "tester" / name; out.mkdir(parents=True, exist_ok=True)
    src = MQL5_DIR / "Files" / f"{name}_report.xlsx"
    MT5MCP().call("tester_get_report", run_id=run_id, path=str(src), file_format="xml")
    shutil.copy(src, out / "report.xlsx")
    return out


def deals(xlsx):
    v = pd.read_excel(xlsx, header=None)
    col0 = v.iloc[:, 0].astype(str).str.strip()
    start = int(np.flatnonzero(col0.to_numpy() == "İşlemler")[0]) + 1
    hdr = [str(x).strip() for x in v.iloc[start].tolist()]
    d = v.iloc[start + 1:].copy(); d.columns = hdr
    d = d[pd.to_datetime(d["Zaman"], errors="coerce").notna()].copy()
    d["Zaman"] = pd.to_datetime(d["Zaman"])
    for c in ("Hacim", "Fiyat", "Komisyon", "Swap", "Kar", "Bakiye"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(" ", "").str.replace(",", "."), errors="coerce")
    return d


def summary_and_chart(d, out, title, deposit=100_000):
    bal = d.set_index("Zaman")["Bakiye"].dropna()
    ex = d[d["Yön"].astype(str).str.lower().isin(["out", "çıkış", "cikis", "exit"])].copy()
    ex["net"] = ex["Kar"].fillna(0) + ex["Komisyon"].fillna(0) + ex["Swap"].fillna(0)
    by_sym = ex.groupby("Sembol").agg(trades=("net", "size"), net=("net", "sum"), win=("net", lambda s: (s > 0).mean())).round(2)
    ent = d[d["Yön"].astype(str).str.lower().isin(["in", "giriş", "giris"])]
    by_leg = ent.groupby(["Sembol", "Yorum"]).size().rename("entries")
    daily = bal.resample("D").last().dropna()
    peak = np.maximum.accumulate(np.maximum(daily.to_numpy(), deposit))
    dd = (peak - daily.to_numpy()) / peak * 100
    yearly = daily.resample("YE").last(); ypnl = yearly.diff().fillna(yearly.iloc[0] - deposit)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(daily.index, daily, color="#2a78d6", lw=1.5, label="Balance (MT5 tester)")
    ax[0].axhline(deposit * 0.92, color="#eb6834", ls="--", lw=1.3, label="Our total floor −8 %")
    ax[0].axhline(deposit * 0.90, color="#e34948", ls=":", lw=1.3, label="FundingPips max loss −10 %")
    ax[0].axhline(deposit * 1.08, color="#008300", ls="--", lw=1.0, label="Phase-1 target +8 %")
    ax[0].set_title(title, loc="left", fontsize=10); ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=0.3)
    ax[1].fill_between(daily.index, dd, color="#2a78d6", alpha=0.35, label="Balance drawdown from peak %")
    ax[1].axhline(8, color="#eb6834", ls="--", lw=1.2, label="8 %"); ax[1].invert_yaxis(); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "balance.png", dpi=100); plt.close(fig)
    by_sym.to_csv(out / "by_symbol.csv"); by_leg.to_csv(out / "entries_by_leg.csv")
    return by_sym, by_leg, {k.year: round(v) for k, v in ypnl.items()}, float(dd.max())


if __name__ == "__main__":
    run_id, name = sys.argv[1], sys.argv[2]
    out = fetch(run_id, name)
    d = deals(out / "report.xlsx")
    s, legs, years, mdd = summary_and_chart(d, out, f"MT5 Strategy Tester — {name}")
    print(s.to_string()); print(legs.to_string()); print("P&L by year:", years, "| max balance DD from peak %:", round(mdd, 2))

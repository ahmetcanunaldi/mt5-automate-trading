"""Forward-test report for the v2.2 EA running on the user's demo account (challenge mode, $100k).

1. Copies <prefix>_entries.csv / <prefix>_deals.csv from Common\\Files (shared by every MT5 terminal of this Windows
   user) to reports/forward/<date>/ and builds the position-level trade history (tools/trade_history.py).
2. Challenge status from closed deals: balance, P&L, static DD from $100k, worst day, Phase-1 (+8 %) / Phase-2 (+5 %)
   progress, trades per leg, rule checks (risk per trade <= 0.5 %, no hedging per symbol, weekend flat).
3. --shadow: runs the same EA in the MT5 Strategy Tester (Vantage data, research reference) over the forward window
   and matches trades by (symbol, leg, direction, entry time +-3 min): the demo must take the same signals.
usage: python tools/forward_report.py [--prefix fwd] [--start 2026-09-28] [--shadow]"""
import argparse
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON = pathlib.Path(r"C:\Users\ahmet\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
DEPOSIT = 100_000.0


def norm_sym(s):
    s = str(s).upper()
    if "XAU" in s or "GOLD" in s:
        return "XAUUSD"
    if any(k in s for k in ("NAS", "US100", "USTEC", "NDX")):
        return "NAS100"
    if any(k in s for k in ("DJ", "US30", "WS30", "DOW")):
        return "DJ30"
    return s


def collect(prefix, out):
    out.mkdir(parents=True, exist_ok=True)
    for k in ("entries", "deals"):
        src = COMMON / f"{prefix}_{k}.csv"
        if not src.exists():
            sys.exit(f"missing {src} - is the EA running with InpExportTrades=true, InpExportPrefix={prefix}?")
        shutil.copy(src, out / f"{k}.csv")
    subprocess.run([sys.executable, str(ROOT / "tools" / "trade_history.py"), str(out)], check=True)
    return pd.read_csv(out / "trade_history.csv", parse_dates=["entry_time", "exit_time"])


def status(P, D):
    closed = P[P.exit_time.notna()].sort_values("exit_time")
    bal = DEPOSIT + closed.net_usd.cumsum()
    dk = closed.exit_time.dt.normalize()
    day = closed.groupby(dk).net_usd.sum()
    day_start = (bal - closed.net_usd).groupby(dk.to_numpy()).first()      # balance before the day's first close
    dd_static = max(0.0, (DEPOSIT - bal.min()) / DEPOSIT * 100) if len(bal) else 0.0
    cur = float(bal.iloc[-1]) if len(bal) else DEPOSIT
    # hedging check: overlapping opposite positions on the same symbol
    hedge = 0
    for s, g in P.groupby(P.symbol.map(norm_sym)):
        g = g.sort_values("entry_time")
        for i, a in g.iterrows():
            ov = g[(g.entry_time < (a.exit_time if pd.notna(a.exit_time) else pd.Timestamp.max)) &
                   (g.exit_time.fillna(pd.Timestamp.max) > a.entry_time) & (g.dir != a.dir)]
            hedge += len(ov)
    wk = P[P.exit_time.notna() & (P.exit_time.dt.dayofweek >= 5)]
    return {"closed_trades": len(closed), "open_positions": int(P.exit_time.isna().sum()),
            "balance": round(cur, 2), "pnl_%": round((cur / DEPOSIT - 1) * 100, 2),
            "phase1_target_%": 8.0, "phase1_progress_%": round(max(0.0, (cur / DEPOSIT - 1) * 100) / 8.0 * 100, 1),
            "max_static_dd_%": round(float(dd_static), 2), "worst_day_%": round(float((day / day_start.to_numpy()).min() * 100), 2) if len(day) else 0.0,
            "max_risk_pct": round(float(P.actual_risk_pct.max()), 3) if len(P) else 0.0,
            "trades_over_0.5%": int((P.actual_risk_pct > 0.5001).sum()), "hedge_overlaps": hedge // 2,
            "weekend_exits": len(wk), "win_rate": round(float((closed.net_usd > 0).mean()), 3) if len(closed) else None,
            "sum_R": round(float(closed.R.sum()), 2)}


def shadow(start, end, prefix="shadow", xau_only=False):
    sys.path.insert(0, str(ROOT / "tools"))
    import tester  # noqa: E402
    p = ["InpInitialBalance=100000", "InpCushionStartPct=2.0", "InpRiskPct=0.5", "InpExportTrades=true", f"InpExportPrefix={prefix}"]
    if xau_only:
        p += ["InpTradeNas=false", "InpTradeDj=false"]
    tester.run("Experts/XauResearch/XauLab.ex5", "forward_shadow", symbol="XAUUSD", model="m1 ohlc", from_date=start,
               to_date=end, deposit=100000, execution_delay=0, params=p)
    return pd.read_csv(COMMON / f"{prefix}_entries.csv", parse_dates=["time"])


def match(P, S):
    a = P.assign(sym=P.symbol.map(norm_sym), t=pd.to_datetime(P.entry_time))[["sym", "leg", "dir", "t"]]
    b = S.assign(sym=S.symbol.map(norm_sym), t=S.time,
                 dir=np.where(S.dir > 0, "BUY", "SELL"))[["sym", "leg", "dir", "t"]]
    used, rows = set(), []
    for i, r in b.iterrows():
        c = a[(a.sym == r.sym) & (a.leg == r.leg) & (a.dir == r.dir) & ((a.t - r.t).abs() <= pd.Timedelta(minutes=3))]
        c = c[~c.index.isin(used)]
        rows.append({**r.to_dict(), "matched": len(c) > 0})
        if len(c):
            used.add(c.index[0])
    M = pd.DataFrame(rows)
    extra = a[~a.index.isin(used)]
    return M, extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fwd")
    ap.add_argument("--start", default=None)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--xau-only", action="store_true", help="demo trades gold only (MetaQuotes-Demo: index CFDs disabled)")
    a = ap.parse_args()
    today = pd.Timestamp.today().normalize()
    out = ROOT / "reports" / "forward" / today.strftime("%Y-%m-%d")
    P = collect(a.prefix, out)
    D = pd.read_csv(out / "deals.csv")
    st = status(P, D)
    print("challenge status:", st)
    by_leg = P.groupby([P.symbol.map(norm_sym), "leg"]).agg(n=("R", "size"), R=("R", "sum"), avg_risk=("actual_risk_pct", "mean")).round(3)
    print(by_leg.to_string())
    pd.Series(st).to_csv(out / "status.csv")
    if a.shadow:
        start = a.start or P.entry_time.min().strftime("%Y-%m-%d")
        S = shadow(start, (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), xau_only=a.xau_only)
        M, extra = match(P, S)
        M.to_csv(out / "shadow_match.csv", index=False); extra.to_csv(out / "demo_only.csv", index=False)
        print(f"shadow (tester, Vantage data): {len(M)} signals, matched by the demo {M.matched.mean():.1%}; "
              f"demo-only trades {len(extra)}")
        print(M[~M.matched].to_string(index=False))

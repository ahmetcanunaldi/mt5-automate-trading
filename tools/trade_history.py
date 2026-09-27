"""Build a position-level trade history from the EA export (entries.csv + deals.csv, XauIdxPortfolio InpExportTrades)
-> reports/tester/<name>/trade_history.xlsx (positions, per-leg summary, risk check) and .csv.
usage: python tools/trade_history.py reports/tester/v22_full"""
import pathlib
import sys

import numpy as np
import pandas as pd

REASON = {0: "client", 1: "mobile", 2: "web", 3: "expert (time/news/guard/weekend)", 4: "stop loss", 5: "take profit", 6: "stop out"}

if __name__ == "__main__":
    d = pathlib.Path(sys.argv[1])
    E = pd.read_csv(d / "entries.csv")
    D = pd.read_csv(d / "deals.csv")
    D["time"] = pd.to_datetime(D["time"], format="%Y.%m.%d %H:%M:%S")
    ins = D[D.entry == 0].set_index("position")
    outs = D[D.entry.isin([1, 3])].groupby("position").agg(exit_time=("time", "last"), exit_price=("price", "last"),
                                                            exit_lots=("volume", "sum"), profit=("profit", "sum"),
                                                            commission=("commission", "sum"), swap=("swap", "sum"),
                                                            reason=("reason", "last"))
    P = E.set_index("position").join(ins[["time", "price", "volume"]].rename(columns={"time": "entry_time", "price": "entry_price",
                                                                                      "volume": "entry_lots"}), how="left").join(outs, how="left")
    P["commission"] = P.commission.fillna(0) + ins.commission.reindex(P.index).fillna(0)
    P["net_usd"] = P.profit + P.commission + P.swap
    P["R"] = P.net_usd / P.risk_usd
    P["actual_sl_dist"] = (P.entry_price - P.sl_price).abs()
    contract = np.where(P.symbol == "XAUUSD", 100.0, 1.0)
    P["actual_risk_usd"] = P.entry_lots * P.actual_sl_dist * contract       # index USD; GER40 not traded in v2
    P["actual_risk_pct"] = P.actual_risk_usd / P.balance * 100
    P["exit_reason"] = P.reason.map(REASON)
    P["hold_h"] = (P.exit_time - P.entry_time).dt.total_seconds() / 3600
    cols = ["symbol", "leg", "dir", "entry_time", "entry_price", "sl_price", "actual_sl_dist", "entry_lots", "balance",
            "risk_usd", "risk_pct", "actual_risk_pct", "risk_mult_used", "exit_time", "exit_price", "exit_reason", "hold_h",
            "profit", "commission", "swap", "net_usd", "R"]
    P = P.reset_index()[["position"] + cols].sort_values("entry_time")
    P["dir"] = P["dir"].map({1: "BUY", -1: "SELL"})
    S = P.groupby(["symbol", "leg"]).agg(trades=("R", "size"), win_rate=("net_usd", lambda s: (s > 0).mean()),
                                         avg_R=("R", "mean"), total_R=("R", "sum"), net_usd=("net_usd", "sum"),
                                         median_risk_pct=("actual_risk_pct", "median"), min_risk_pct=("actual_risk_pct", "min"),
                                         max_risk_pct=("actual_risk_pct", "max"), risk_mult=("risk_mult_used", "median")).round(3)
    chk = pd.DataFrame({"check": ["positions", "max actual risk % of balance", "positions with actual risk > 0.5 %",
                                  "median actual risk % (full-size legs)", "median actual risk % (season, half size)",
                                  "positions without exit (open at end)"],
                        "value": [len(P), round(P.actual_risk_pct.max(), 4), int((P.actual_risk_pct > 0.5 + 1e-6).sum()),
                                  round(P[P.risk_mult_used >= 0.999].actual_risk_pct.median(), 4),
                                  round(P[P.leg == "season"].actual_risk_pct.median(), 4), int(P.exit_time.isna().sum())]})
    with pd.ExcelWriter(d / "trade_history.xlsx") as w:
        P.round(4).to_excel(w, sheet_name="positions", index=False); S.to_excel(w, sheet_name="by_leg"); chk.to_excel(w, sheet_name="risk_check", index=False)
    P.to_csv(d / "trade_history.csv", index=False)
    pd.set_option("display.width", 250)
    print(chk.to_string(index=False)); print(S.to_string())
    x = P[(P.symbol == "XAUUSD") & (P.entry_time >= "2025-01-02") & (P.entry_time < "2025-01-10")]
    print(x[["entry_time", "leg", "entry_price", "sl_price", "actual_sl_dist", "entry_lots", "balance", "actual_risk_pct", "net_usd", "R"]].to_string(index=False))

"""Fetch long H1 / D1 history through the MCP get_chart_history tool (yearly chunks) -> data/<SYM>_<TF>_hist.parquet.
Symbols not in Market Watch are added for the download and removed afterwards (data access only, no trading).
usage: python tools/fetch_history.py H1 2010 EURUSD GBPUSD ... [--temp AUDNZD]"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd  # noqa: E402

from mcp_client import MT5MCP  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def fetch(c, sym, tf, first_year, last="2026-09-26"):
    parts = []
    for y in range(first_year, int(last[:4]) + 1):
        a, b = f"{y}-01-01T00:00:00", (f"{y + 1}-01-01T00:00:00" if y < int(last[:4]) else f"{last}T00:00:00")
        r = c.call("get_chart_history", symbol=sym, period=tf, datetime_from=a, datetime_to=b)
        if r.get("history"):
            parts.append(pd.DataFrame(r["history"]))
    d = pd.concat(parts)
    d["time"] = pd.to_datetime(d["time"])
    d = d.set_index("time").sort_index()
    return d[~d.index.duplicated()]


if __name__ == "__main__":
    tf, y0 = sys.argv[1], int(sys.argv[2])
    args = sys.argv[3:]
    temp = args[args.index("--temp") + 1:] if "--temp" in args else []
    syms = (args[:args.index("--temp")] if "--temp" in args else args) + temp
    c = MT5MCP()
    for s in temp:
        c.call("add_marketwatch_symbol", symbol=s, show=True)
    try:
        for s in syms:
            d = fetch(c, s, tf, y0)
            d.to_parquet(DATA / f"{s}_{tf}_hist.parquet")
            print(s, tf, len(d), d.index[0], "->", d.index[-1], flush=True)
    finally:
        for s in temp:
            c.call("remove_marketwatch_symbol", symbol=s)

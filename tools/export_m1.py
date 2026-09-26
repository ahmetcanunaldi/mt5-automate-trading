"""Export full M1 history of symbols through the Strategy Tester (bypasses the 100k max-bars cap).
usage: python tools/export_m1.py XAGUSD EURUSD USDX.r"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from deploy import MQL5_DIR  # noqa: E402
from mcp_client import MT5MCP  # noqa: E402

COMMON = pathlib.Path(r"C:\Users\ahmet\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

if __name__ == "__main__":
    c = MT5MCP()
    ex5 = str(MQL5_DIR / "Experts/XauResearch/DataExporter.ex5")
    for sym in sys.argv[1:]:
        ini = str(MQL5_DIR / f"Profiles/Tester/export_{sym}.ini")
        c.call("tester_prepare_config", mql5_program_path=ex5, symbol=sym, model="m1 ohlc", timeframe="M1",
               from_date="2018-01-01T00:00:00", to_date="2026-09-26T00:00:00", deposit=10000,
               deposit_currency="USD", leverage=100, execution_delay=0, optimization=False, output_path=ini)
        rid = c.call("tester_run_backtest", config_path=ini, wait=False)["run_id"]
        t0 = time.time()
        while True:
            c.call("tester_wait", run_id=rid, timeout_sec=60)
            st = c.call("tester_get_status", run_id=rid)
            if str(st.get("tester_status", "")).lower() == "stopped" or time.time() - t0 > 3600:
                break
        f = COMMON / f"xau_export_{sym}_M1.csv"
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").sort_index()
        df = df[~df.index.duplicated()]
        df.to_parquet(DATA / f"{sym.replace('.r', '')}_M1_2018.parquet")
        print(sym, len(df), df.index[0], "->", df.index[-1], f"{time.time()-t0:.0f}s", flush=True)

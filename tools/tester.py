"""Helpers to run MT5 Strategy Tester jobs through the MCP server."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mcp_client import MT5MCP  # noqa: E402
from deploy import MQL5_DIR  # noqa: E402

PROFILES = MQL5_DIR / "Profiles" / "Tester"


def run(ex5_rel, name, symbol="XAUUSD", model="real ticks", timeframe="M1",
        from_date="2025-01-01", to_date="2026-09-26", deposit=10000, leverage=100,
        params=None, execution_delay=-1, wait_sec=3600, client=None, report_path=None):
    c = client or MT5MCP()
    ex5 = str(MQL5_DIR / ex5_rel)
    ini = str(PROFILES / f"{name}.ini")
    c.call("tester_prepare_config", mql5_program_path=ex5, symbol=symbol, model=model, timeframe=timeframe,
           from_date=from_date + "T00:00:00", to_date=to_date + "T00:00:00", deposit=deposit,
           deposit_currency="USD", leverage=leverage, execution_delay=execution_delay,
           optimization=False, output_path=ini)
    kw = {}
    if params:
        setp = str(PROFILES / f"{name}.set")
        c.call("tester_prepare_inputs", mql5_program_path=ex5, parameters=params, output_path=setp)
        kw["inputs_path"] = setp
    started = c.call("tester_run_backtest", config_path=ini, wait=False, **kw)
    run_id = started["run_id"] if isinstance(started, dict) else started
    t0 = time.time()
    while time.time() - t0 < wait_sec:
        st = c.call("tester_wait", run_id=run_id, timeout_sec=60)
        status = c.call("tester_get_status", run_id=run_id)
        print(f"[{int(time.time()-t0)}s] {status}", flush=True)
        if isinstance(status, dict) and str(status.get("status", "")).lower() in ("stopped", "finished", "completed"):
            break
    rep = c.call("tester_get_report", run_id=run_id, **({"path": report_path} if report_path else {}))
    return run_id, rep

"""Sync repo mql5/ sources into the terminal's MQL5 folder (via MCP) and compile with MetaEditor CLI.

usage: python tools/deploy.py Experts/XauResearch/DataExporter.mq5 [more targets...]
All .mq5/.mqh files under mql5/ are synced; only the given targets are compiled.
"""
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mcp_client import MT5MCP, ROOT  # noqa: E402

TERMINAL_DIR = pathlib.Path(r"D:\copy-trade-web-application\mt5_slave_2")
MQL5_DIR = pathlib.Path(r"C:\Users\ahmet\AppData\Roaming\MetaQuotes\Terminal\8A8C66D52CAAD4C0287F6DD96312D6F6\MQL5")
SRC = ROOT / "mql5"


def sync(client):
    for f in SRC.rglob("*"):
        if f.suffix in (".mq5", ".mqh") and f.is_file():
            dst = MQL5_DIR / f.relative_to(SRC)
            client.call("write_file", path=str(dst), content=f.read_text(encoding="utf-8"), overwrite=True)


def compile_one(rel):
    target = MQL5_DIR / rel
    log = ROOT / "reports" / "compile.log"
    subprocess.run([str(TERMINAL_DIR / "MetaEditor64.exe"), f"/compile:{target}", f"/log:{log}"], check=False)
    text = log.read_text(encoding="utf-16", errors="replace") if log.exists() else ""
    tail = [ln for ln in text.splitlines() if ln.strip()][-3:]
    print(f"[{rel}]", *tail, sep="\n  ")
    return "0 errors" in text.lower() or "0 error" in text.lower()


if __name__ == "__main__":
    sync(MT5MCP())
    ok = all(compile_one(t) for t in sys.argv[1:])
    sys.exit(0 if ok else 1)

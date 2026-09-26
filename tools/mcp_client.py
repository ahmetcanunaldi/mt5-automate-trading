"""Minimal client for the MetaTrader 5 MCP server (streamable HTTP, JSON responses).

Safety: this project only uses data / file / tester / journal / calendar tools.
Trade tools (trade_*) are blocked here on purpose - the terminal is a REAL account.
"""
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCKED_PREFIX = ("trade_",)


def _load_cfg():
    lines = (ROOT / "mt5-mcp.txt").read_text(encoding="utf-8").splitlines()
    url = lines[0].split(":", 1)[1].strip()
    key = lines[1].split(":", 1)[1].strip()
    return url, key


class MT5MCP:
    def __init__(self, timeout=600):
        self.url, self.key = _load_cfg()
        self.timeout = timeout
        self.session = None
        self._id = 0
        self._init()

    def _post(self, payload, want_headers=False):
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        req.add_header("Authorization", f"Bearer {self.key}")
        if self.session:
            req.add_header("Mcp-Session-Id", self.session)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if want_headers:
                return resp.headers, body
            return body

    def _init(self):
        self._id += 1
        headers, _ = self._post({"jsonrpc": "2.0", "id": self._id, "method": "initialize",
                                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                            "clientInfo": {"name": "xau-research", "version": "0.1"}}},
                                want_headers=True)
        self.session = headers.get("Mcp-Session-Id")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.call("get_workspace_info")  # mandatory pre-flight

    def call(self, name, **args):
        if name.startswith(_BLOCKED_PREFIX):
            raise PermissionError(f"{name} is blocked: terminal is a REAL account (data/backtest only)")
        self._id += 1
        body = self._post({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                           "params": {"name": name, "arguments": args}})
        d = json.loads(body)
        if "error" in d:
            raise RuntimeError(d["error"])
        res = d["result"]
        text = "".join(c.get("text", "") for c in res.get("content", []))
        if res.get("isError"):
            raise RuntimeError(text[:2000])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

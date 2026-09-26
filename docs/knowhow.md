# Know-how (distilled, keep updated)

## Environment / data
- MCP terminal = `D:\copy-trade-web-application\mt5_slave_2`, Vantage REAL account (balance 0). **Never use trade tools.** `tools/mcp_client.py` blocks `trade_*`.
- MCP file tools need **absolute paths** under the MQL5 data folder.
- MetaEditor CLI compile works (`tools/deploy.py`), but the running terminal does **not** index newly compiled EAs until Navigator → Refresh (or restart). Tester MCP rejects un-indexed programs ("must reference a compiled program"). Recompiling an already-indexed EA is fine.
- "Max bars in chart" = 100k caps `copy_rates_*`: H1 from 2007, M15 from 2022-07, M5 from 2025-04, M1 ~3 months. `copy_rates_range/from` silently return 1 bar outside the window → use `copy_rates_from_pos`.
- Ticks are not capped: `copy_ticks_from` gives full history from 2024-12-04; M1 bars rebuilt from ticks match terminal M1 exactly (`data/XAUUSD_M1_from_ticks.parquet`, includes mean/max spread).
- Python install has `python311._pth` → cwd/PYTHONPATH ignored. Scripts insert repo root into `sys.path`; tests use root `conftest.py`.
- pandas 3 parquet timestamps are `datetime64[ms]` — never assume ns (`asi8` bug found in EXP-001 dry run).

## Market microstructure (Vantage XAUUSD, 2024-12..2026-09)
- Server time = NY-close convention: UTC+2 winter / UTC+3 US-DST. Daily bars 01:00–23:59 server. London open ≈ 10:00, US data 15:30, NY cash open 16:30 server (all seasons).
- MT5 economic-calendar timestamps are a **fixed UTC+3** (not server time): winter server = calendar − 1h, summer = calendar.
- Mean spread ≈ 20 pts ($0.20), rising from ~16 (2024-12) to ~23 (2026). Flat across the session except around rollover.
- Median daily range ≈ $72 at ~$4,000+ gold → a $6 target is ~8 % of a day's range.

## Edges found so far (DEV 2022-07..2024-11, VAL 2024-12..2025-09)
- **US-session breakouts in the H1 trend direction** are the most robust structure (NY ORB 16:30 30-min range; Donchian-32 on M15 after 15:00). Removing the H1 trend filter (EMA50 vs EMA200 + close vs EMA50) kills the edge.
- Breakouts need room: trailing at 1–1.5 ATR destroys them; 3 ATR or no trailing with 2–3 R targets and ≥ 4 h max hold works.
- Mean reversion to VWAP worked in the 2022–24 range regime but lost heavily in the 2025 trend regime — avoid unless regime-gated.
- Raw time-of-day / session drifts are real but tiny (1–4 bps) versus the $6 minimum target.

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
- Broker H1 history before 2018 contains one bar per day (not intraday) → intraday research can only start 2018.
- Coarse-bar proxies (H1 exec with SL-first) are not trustworthy for breakout legs: sign flips vs M15/M1 execution.

## Statistical caution
- ~1,200 configurations were tried in EXP-003/007. With that many (correlated) trials the expected best DEV Sharpe under
  the null is ~2 (annualized, ~600 days). DEV Sharpe alone is therefore NOT evidence; VAL (independent 10 months)
  SR 2.34 on 189 trades (t ≈ 2.1) is the real, still modest, evidence. OOS (2025-10..2026-09) stays locked.

## OOS lesson (EXP-013, 2025-10..2026-09)
- The US-session H1-trend breakout portfolio (DEV SR 1.75 / VAL 2.35) dropped to SR 0.13 (conservative) and 0.56 (aggressive K4)
  in the one-shot OOS. The regime changed: volatility 3–4×, blow-off top and crash, spreads +30 %. Shorts lost, longs still won.
- Risk engine worked: no daily or static-total breach even in the bad regime (worst static DD ≈ 4.7 %, worst day 1.9 %).
- Concurrency (K4, 2 % open risk) raises speed but peak-to-trough DD reached 10.3 % in OOS → too aggressive for an 8 % peak-DD reading.
- Next research must be walk-forward (re-fit every N months on trailing data) over 2022-07..2026-09 and regime-aware
  (volatility-scaled targets, regime switch), then confirmed on a live demo — there is no untouched history left.

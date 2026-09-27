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

## ML / price-action findings (EXP-014..016, tick era 2024-12..2026-09)
- Beware "TP-first" labels with a time barrier: timeouts labelled 0 let the model score AUC 0.6+ by predicting volatility
  (hour/news/ATR), not direction. Always test with a pure-direction label (timeouts dropped) and look at realized R by decile.
- Pure direction AUC ≈ 0.51 with 120 features (3-scale swing structure, S/R clusters, tick microstructure, 6 cross assets).
- 27 price-action setups × 5 barrier schemes: none significant after costs; signs flip between half-years.
- Costs matter: baseline mean R per random entry is −0.03…−0.07 R (spread 0.2$, slip, $7/lot) — any edge must beat that.

## Long history (tester export, EXP-020)
- After Navigator → Refresh, the MCP tester runs our EAs. DataExporter in "m1 ohlc" mode exports full M1 history
  (2018-09..) to Common\Files in ~5 min — bypasses the 100k "max bars" cap. File: data/XAUUSD_M1_2018.parquet.
- The rule portfolio (EXP-005/013) is regime-specific: negative in 2019–2021 (−38 R), positive 2022–2025 (+94 R), mixed 2026.
  Any future candidate must be judged on 2019–2026 walk-forward, not on a single era.
- Position-size floor: $10k × 0.5 % = $50 risk; at 0.01 lot a $1 move = $1 → maximum stop ≈ $49 (after commission).
  In 2026 (daily ATR $80–226) any strategy with daily-ATR-scale stops cannot be traded on this account size.
- 8-year verdict (EXP-022..024): no intraday horizon (M5 → end of day) shows a cost-beating, year-stable edge in
  price-action, supply/demand, ML direction (AUC 0.51–0.52 every year) or daily momentum/cross-asset signals.

## Tester / EA notes (EXP-050)
- New EA files are not indexed until Navigator → Refresh. Workaround used: compile experiments into the already
  indexed slot `Experts/XauResearch/XauLab` (copy of the canonical source, e.g. `XauScalper/XauPortfolio.mq5`).
- "1 minute OHLC" tester runs of 5+ years take ~20 s; "every tick" ~2 min; "real ticks" currently fails with
  "tick cache error 7".
- 2026-09-27 01:37: the user's Python 3.11/3.12 installations were deleted by something outside this session
  (not by our commands). The research stack needs Python 3.11+ with pandas, numpy, numba, lightgbm, pyarrow, MetaTrader5.

## Current best (2026-09-27): 10-leg portfolio = EXP-047 legs + tday900 (EXP-052/053)
- Legs: H4 Donchian-180 long (trail 6 ATR), trend-day continuation 15:00 & 18:00, Asia-open drift (Tue–Fri),
  Friday long, Friday-close drift, turn-of-month long, Larry-Williams breakout (k0.4 from 10:00), inside-day & NR7
  breakouts with D1 trend. All 0.5 % risk, ≤ 6 positions, one direction, weekend flat, news rules, FP guards.
- 2019–26: SR ≈ 1.2, ≈ 18 R/yr, peak DD ≈ 8 %, worst day 2.7 %, MC95 8.2 %, every year positive, 0 % breach,
  P1+P2 pass ≤ 250 days ≈ 35–37 %, median ≈ 175 days. 2021–26: SR ≈ 1.56.
- MT5 tester parity checked on the 9-leg EA (XauPortfolio.mq5): 3,111 vs 3,101 trades, equity DD 7.7 vs 7.9 %.
- Gold structure learned: continuation/momentum weakly positive at every horizon ≥ 30 min, mean reversion negative;
  calendar effects (Asia-open drift, Friday, Friday close, turn-of-month, post-holiday) persist 2008–2026.

## Current best (update 2026-09-27, EXP-069): "best13"
- best10 legs (EXP-047 + tday 15:00) + strong-close continuation (clv > 0.6 → next-day long) + Asia drift only on
  low-vol days + Jan/Jul/Aug season leg at half risk + trailing 1.5× stop on intraday legs.
- 2019–26: +$138.7k on $100k (11.9 %/yr), SR 1.53, peak DD 8.6 %, worst day 2.1 %, every year positive,
  0.69 R/week, P1+P2 within 250 d in 54 % of start dates, 0 % breaches.
- Strong-close edge: persists 2008–2026 incl. the 2013–18 bear; realised mostly in Asian hours but holding to the
  close is still better after costs. Weakness does NOT persist (no short counterpart).

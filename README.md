# XAUUSD intraday algo — FundingPips 2-Step Standard $10k

- `docs/rules.md` — prop-firm + internal rules. `docs/knowhow.md` — distilled learnings. `EXPERIMENTS.md` — every experiment.
- `research/` — Python lab: `data_loader.py` (MT5 → parquet), `engine.py` (numba backtester with guards),
  `calendar_news.py`, `features.py`, `metrics.py` (Sharpe, DD, Monte Carlo, challenge sim, gates), `lab.py`, `strategies/`.
- `mql5/` — EA sources, synced + compiled into the terminal by `tools/deploy.py`; tests run via `tools/tester.py` (MCP).

```
python research/data_loader.py --bars H1 M15 M5 M1 --ticks   # refresh data
python -m pytest -q research/tests                            # engine / guard tests
python research/run_baseline.py                               # EXP-001
```

# Experiment log

Every experiment gets an ID, is reproducible from the repo, and has its raw output in `reports/EXP-###/`.
Periods: DEV 2022-07-05..2024-11-30 (M15 exec, pessimistic), VAL 2024-12-04..2025-09-30 (M1 exec),
OOS 2025-10-01..2026-09-25 (M1 exec, **locked** until a final candidate). Costs: real spread (floor 15 pts),
$7/lot commission, 5 pts slippage per side. Guards per docs/rules.md. Fresh $10k per period.

| ID | Date | Hypothesis / change | Result (DEV / VAL) | Verdict | Lesson |
|---|---|---|---|---|---|
| EXP-000 | 2026-09-26 | Infrastructure: engine, guards, news filter, metrics, challenge sim; 9 unit tests | tests pass | ✅ | pandas 3 ms-resolution timestamps broke day ids — always cast |
| EXP-001 | 2026-09-26 | Baseline: 5 rule families (Asia breakout, NY ORB, EMA pullback, VWAP reversion, prev-day break), default params, TP ≥ $6 | all PF 0.67–1.11; best VAL: NY ORB SR 0.69, prev-day SR 0.59; DEV all negative except VWAP rev (SR 0.22) | ❌ | Naive textbook rules have no edge after costs on 2022–25 gold; need data-driven edge discovery first |

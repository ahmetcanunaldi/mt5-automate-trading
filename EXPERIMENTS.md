# Experiment log

Every experiment gets an ID, is reproducible from the repo, and has its raw output in `reports/EXP-###/`.
Periods: DEV 2022-07-05..2024-11-30 (M15 exec, pessimistic), VAL 2024-12-04..2025-09-30 (M1 exec),
OOS 2025-10-01..2026-09-25 (M1 exec, **locked** until a final candidate). Costs: real spread (floor 15 pts),
$7/lot commission, 5 pts slippage per side. Guards per docs/rules.md. Fresh $10k per period.

| ID | Date | Hypothesis / change | Result (DEV / VAL) | Verdict | Lesson |
|---|---|---|---|---|---|
| EXP-000 | 2026-09-26 | Infrastructure: engine, guards, news filter, metrics, challenge sim; 9 unit tests | tests pass | ✅ | pandas 3 ms-resolution timestamps broke day ids — always cast |
| EXP-001 | 2026-09-26 | Baseline: 5 rule families (Asia breakout, NY ORB, EMA pullback, VWAP reversion, prev-day break), default params, TP ≥ $6 | all PF 0.67–1.11; best VAL: NY ORB SR 0.69, prev-day SR 0.59; DEV all negative except VWAP rev (SR 0.22) | ❌ | Naive textbook rules have no edge after costs on 2022–25 gold; need data-driven edge discovery first |
| EXP-002 | 2026-09-26 | Stat edge discovery: hour drift (H1 2010–26), session→session predictability, post-news drift, DoW, M15 autocorr (`research/explore_stats.py`, `reports/EXP-002_stats.txt`) | 01:00 hour +3 bps (t=5); Asia→NY-AM same sign 55 % (top-30 % Asia moves); post-news continuation 52.7 %; London hours (10–12) M15 lag-1 autocorr −0.08…−0.11 | ℹ️ | All raw edges are 1–4 bps (≈$0.5–1.6) ≪ $6 target → need trend/breakout structures that let winners run |
| EXP-003 | 2026-09-26 | Random search, 6 families × ~150 configs (897 total), exits: rr/trail/BE/hold; select on DEV, check VAL (`research/sweep.py`) | Consistent: NY ORB+trend (DEV SR 1.46 / VAL 2.29), Donchian-32 US session+trend (1.17 / 2.14), prev-day break long-only (1.22 / 1.23). VWAP reversion: DEV 0.84 → VAL −2.0 | ✅ lead | Edge = **US-session breakouts in H1-trend direction**. Mean reversion breaks in 2025 trend regime. 897 trials → selection bias, use plateaus |
| EXP-004 | 2026-09-26 | One-at-a-time sensitivity around the 3 leads (`research/robustness.py`) | NY ORB: plateau over sl 1.5–3 ATR, rr 1–5, hold ≥ 240; dies without trend filter (SR ≈ 0) or with or_start 15:30. Donchian: plateau n 32–48, sl 1.5–3, rr 2–5, start 15:00 essential. Trail 1–1.5 ATR hurts everywhere | ✅ | H1 trend filter + US session are the load-bearing parts; tight trailing kills breakouts |
| EXP-005 | 2026-09-26 | Single-position portfolio of central (not peak) params (`research/portfolio.py`) | ny_orb+donchian_us+pdb_long: DEV n=526 SR 1.70 PF 1.34 DD 5.6 % mc95 8.9 %; VAL n=189 SR 2.34 PF 1.51 DD 2.8 %. Shorts also positive (0.09R / 0.24R). Every year positive. Challenge pass (P1+P2) 20 % DEV / 27 % VAL within 120 days — too slow (~13 %/yr) | ⚠️ close | Meets Sharpe/DD on DEV+VAL but MC95 DD slightly > 8 % and returns too slow for a practical challenge → improve expectancy per trade (ML filter) and add uncorrelated legs |

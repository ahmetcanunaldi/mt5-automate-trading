# Quant-math branch — distilled findings (branch `quant-math`)

Scope: only price / volume / history, model-based inference (SDEs, filtering, optimal control / portfolio),
strict validation (walk-forward, CSCV-PBO, deflated Sharpe, SPA, placebo, drift benchmark).
DEV data 2018-01..2024-12; **lockbox 2025-01..2026-09 untouched** until phase 7.

## Phase 1 — predictability map (QM-001)
- Direction is close to a random walk: DFA Hurst 0.46–0.52 for every symbol and bar size (1 min .. 1 day).
- Short-horizon negative autocorrelation (bid-ask bounce / microstructure): XAU, XAG, EURUSD at 1–15 min,
  rho1 −0.02, t −3…−9 — highly significant but worth 2–8 % of the round-trip cost → not tradeable.
- **Equity indices mean-revert at the daily scale**: rho1(daily) ≈ −0.11 for NAS100, DJ30, SP500, GER40
  (NAS t −2.5 robust; others −1.6…−1.9); implied edge 7–9× cost. Also 4h rho −0.04…−0.07 (edge ≈ 1–2× cost).
- Cross-asset lead-lag at 1 min is ≤ 0.02 correlation (e.g. SP500 → GER40 +0.02) — far below cost.
- Mutual information is large at 1 min but comes from magnitude (volatility clustering), not sign.
- ⇒ Priorities: volatility forecasting (phase 2) and daily-scale mean reversion / OU on indices (phases 3–4).

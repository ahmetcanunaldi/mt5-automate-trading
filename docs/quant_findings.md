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

## Phase 2 — volatility (QM-002, QM-003)
- HAR-RV with leverage + jump terms is the best one-day-ahead variance forecaster on all 8 symbols (log R² 0.45–0.75);
  GARCH on daily returns is clearly worse.
- Rough volatility confirmed: Hurst of log-volatility 0.10–0.19 (literature ~0.1).
- Volatility-managed sizing (≤ 0.5 % cap, so only scaling down) improves gold (SR 0.85 → 0.97, DD −25 %,
  placebo p 0.00) but not equity indices. Portfolio v2: +0.04 SR on DEV.

## Phase 3/4 lead check — index daily AR (QM-004)
- After standardizing by forecast volatility, daily index autocorrelation is ≈ 0 and unstable; the raw −0.11 came
  from crisis days. No timing value beyond the drift (placebo p ≥ 0.2).

## Phase 3 — regimes (QM-006)
- HMM (2/3 states) and Kalman drift filters do not beat always-long on any symbol; HMM states are volatility
  regimes; the Kalman drift signal-to-noise ratio is so low that the filtered drift never changes sign on indices.

## Phase 4 — cointegration / OU (QM-005)
- Index pairs and XAU–XAG are cointegrated in only 14–22 % of rolling windows; Kalman–OU–Bertram stat-arb loses
  after costs on H1 and is insignificant on D1 (placebo p ≥ 0.23).

## Phase 5 — jumps, Hawkes, volume (QM-007)
- No post-jump drift; jumps mildly self-exciting (Hawkes branching 0.10–0.23); CGW volume effect has the textbook
  sign (high-volume moves revert) but is tiny. Order-flow tests need tick data that exist only in the lockbox.

## Phase 6 — optimal control (QM-008, QM-009) — the useful result
- The prop challenge is a goal-reaching problem: with a positive edge and no time limit, P(success) rises as size
  falls; the price is time. v2 at 0.5 %: P(pass) ≈ 97 %, fail ≈ 2–3 %, ~256 days (bootstrap).
- **Cushion-proportional sizing** (risk × (8 % − DD)/(8 % − 2 %), floor 0.1) keeps the upside and cuts the failure
  probability from 2.2 % to 0.03 % (challenge) and the 2-year funded loss probability from 5.6 % to 0.03 %, at
  ~5 extra days / −0.12 payouts per year. It never bound on the real 2019–24 path (no historical cost).

## Round 2 (QM-010 … QM-013)
- Realized skewness / kurtosis / signed jump variation carry no directional information; a naive pooled z of −5
  on 5-day targets was fully reproduced by the placebo (overlapping targets + refitted slopes) → artefact.
- Volatility surprises do not make intraday direction persistent.
- Local memory is not persistent: consecutive-window VR / autocorrelation are uncorrelated → Hurst / VR regime
  switching cannot work.
- Diversified vol-scaled TSMOM (7 assets, 2015–24) SR ≤ 0.38 < long-only risk parity 0.78.

## Bottom line after 13 model families
Using only price, volume and history, **no directional model survives** walk-forward + placebo + drift benchmark
+ costs on XAU, XAG, NAS100, DJ30, SP500, GER40, EURUSD, USDJPY. What mathematics delivers:
1. volatility forecasts (HAR, rough vol) → vol-managed sizing on gold (SR 0.85 → 0.97, placebo p 0.00);
2. optimal risk control for the prop objective → cushion-proportional sizing (failure 2.2 % → 0.03 %).
Candidates for the one-shot lockbox test: (1) cushion rule, (2) gold vol-managed sizing.

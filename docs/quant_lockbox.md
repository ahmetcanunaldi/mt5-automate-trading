# Quant-math lockbox — pre-registration (written BEFORE opening 2025-01-01 .. 2026-09-25)

Opened once, for exactly these two candidates, with exactly these settings. No parameter changes after opening.

Caveat stated in advance: the v2 legs themselves were researched on data through 2026-09 (main branch), so the
v2 book is NOT out-of-sample in 2025–26. The lockbox is clean only for the quant-branch risk candidates below,
whose parameters were fixed on DEV 2018–2024 (QM-003, QM-008/009).

## Candidate 1 — cushion-proportional sizing (QM-008/009)
Rule: risk × clip((8 − static DD%) / (8 − 2), 0.1, 1) (`Guards.cushion_start_pct = 2.0`), replacing the step rule
(half size below 6.5 % DD). Evaluated on v2 (XAU best13 + NAS100/DJ30 prior legs, K6, open risk 3 %, news v2).
PASS if all hold on the lockbox:
- real path, challenge mode: net profit and Sharpe within 5 % of the step rule (no material cost), peak DD not higher;
- real path, funded mode (+3 % payouts, 35 % consistency, 1.25 % cap): payouts within 1 of the step rule, no breach;
- block-bootstrap MC of the lockbox daily R: P(fail challenge) and P(funded account lost in 2 y) not higher than
  the step rule's.

## Candidate 2 — gold volatility-managed sizing (QM-003)
Rule: XAU trades risk × min(1, (σ_ref / σ̂)^2), floor 0.2; σ̂ = HAR(+leverage+jump) one-day-ahead RV forecast
(walk-forward, refit each January), σ_ref = trailing 250-day median of σ̂.
PASS if all hold on the lockbox:
- XAU always-long (01:05–23:30, SL 1 ATR_D, trail 1.5×): Sharpe(managed) > Sharpe(unmanaged) and
  placebo p ≤ 0.10 (multiplier shuffled within years, 20×);
- v2 with the rule on its XAU legs: Sharpe not lower than v2 without it, peak DD not higher.

## Results (opened once, 2026-09-27, `research/quant/qm_lockbox.py`, reports/quant/LOCKBOX/result.txt)
Lockbox regime: exceptionally favourable — v2 daily R mean 0.221 (DEV 0.094), sd 1.10; v2 SR 3.2 (v2 legs are not
out-of-sample here).

**Candidate 1 — cushion sizing: PASS.** Real path identical to the step rule (the rule never bound: +63.2 %, SR 3.20,
DD 4.32 %, 10 funded payouts, no breach, both). Bootstrap MC of lockbox R: P(fail challenge) 0.075 % → 0 %,
P(funded account lost in 2 y) 0.45 % → 0 %, payouts/yr 7.73 → 7.64. Weak stress test (benign period) but no cost
and the predicted direction.

**Candidate 2 — gold vol-managed sizing: FAIL.** XAU always-long SR 1.28 → 1.15 (placebo p 0.45); v2 net +63 % →
+51 %, SR 3.20 → 3.10. In 2025–26 gold's volatility and returns rose together (the blow-off rally), so cutting size
in high volatility cut the best days. The DEV result (QM-003) did not generalize.

# FX majors breadth search ("act like Renaissance") — findings (FXR-001 … 003, 2026-09-27)

Universe: USD, EUR, GBP, JPY, CHF, AUD, CAD via 6 majors (NZD excluded: broker history of NZDUSD / AUDNZD starts
2026). H1 2015–2024 DEV; 2025–26 holdout left unopened (no candidate reached it). Costs 1.0–3.7 bp round turn
(recorded Vantage spreads, $5/lot, 0.3 pip slippage per side) — conclusions also hold on gross returns.

| Family | Best result | Verdict |
|---|---|---|
| Cross-sectional / time-series momentum, reversal, MA gap, PCA residuals, low vol, last-session (18 signals) | XS IC ≤ 0.016 (t 1.65), gross SR ≤ 0.28 | none |
| Carry (FRED 3-month rates) | net SR −0.41 (2015–24), positive only 2023–24 | none |
| Home-session depreciation (Breedon–Ranaldo) | 0.01 bp per trade gross | none |
| Month-end hedge rebalancing at the 4 pm fix (Melvin–Prins) | wrong sign, unstable | none |
| Pooled ridge / LightGBM on 34 features, walk-forward | IC ≈ 0, placebo p 1.0 | none |

Why Renaissance can and we cannot: IR ≈ IC·√breadth. Their breadth is thousands of instruments × many trades per day
at ~0.1 bp cost; ours here is ~7 near-dependent currency bets per day at 1–4 bp. An IC of 0.01–0.02, which pays them,
is worth less than our spread. With these instruments and costs, a daily FX signal needs an IC of ≈ 0.05 net of
cost; nothing in price, rates, cross-asset or calendar data reaches it on G7 in 2015–24.

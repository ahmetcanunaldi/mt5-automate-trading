# Algorithmic trading literature 2025–2026 — scan and relevance (2026-09-27)

| Topic | What the 2025–26 literature says | Relevance to this project |
|---|---|---|
| Time-series foundation models (Chronos/Chronos-2, TimesFM-2.5, Moirai-2.0, TimeGPT; finance-native **Kronos**, 12 B OHLCV bars) | Zero-shot daily return forecasting is weak: Chronos R² −1.4 %, 51 % direction; TimesFM ≈ 50 % (Re(Visiting) TSFM in Finance). Gains over the random walk are "small and sparse"; Diebold-Mariano beats RW in 2 of 10 tasks; "not universal engines for statistically reliable alpha". Finance-native pretraining helps; CatBoost/LightGBM remain the strongest baselines. Kronos reports RankIC / volatility / generation gains, no after-cost trading. | Matches our ML result (AUC 0.51). Cross-sectional stock ranking ≠ single-asset timing. |
| TSFMs for volatility | Vs Log-HAR on 50 assets: only Tiny Time Mixers beats HAR, narrowly; equal-weight TTM + HAR is as good as any model and more robust. | Our HAR choice (QM-002) is state of the art; a TTM+HAR ensemble is the only upgrade path (small). |
| LLM trading agents (Agent Market Arena, BacktestBench, CLQT, Backtrader-Bench, agentic-trading survey) | Benchmarks are young; live windows of weeks; agent architecture matters more than the LLM; cumulative return conflates drift, style tilts and skill. | No robust evidence of alpha; confirms our beta-vs-timing lesson (EXP-099). |
| Deep reinforcement learning | Critical survey of 167 papers: 86 % claim outperformance without statistical tests, 89.5 % ignore realistic costs; open problems: non-stationarity, sim-to-real gap. | Published RL results are mostly not trustworthy evidence. |
| "Virtue of complexity" debate | 2025 rebuttals (Buncic; Stockholm; Cartea–Jin–Shi "limited virtue in a noisy world"): a shrunk expanding-window linear model beats the complex models (SR 0.70 vs 0.49). | Simple + shrinkage ≥ complex for market timing — as in our tests. |
| Trend following | CTAs lost ~9 % YTD by Apr/Aug 2025 (SG Trend, TTU). "Is Trend Still Your Friend?" (2026): short-term trends have not paid since ~2009 on **small-tick** contracts (vol-normalised tick size); HFT market makers break the impact feedback loop; large-tick contracts keep trend profits. | Explains our QM-013 / FX swing failures: XAU and index CFDs are small-tick. |
| Volatility-managed portfolios | Over 103 strategies no systematic Sharpe gain out of sample; real-time implementation fails (Cederburg et al.); post-2003 gain ≈ 0.02. | Exactly our lockbox result (gold vol sizing FAIL). |
| Alpha decay / crowding | ~50 % of anomaly alpha disappears after publication; "mechanical" signals (momentum) crowd hyperbolically; crowding accelerated after 2015; AI-driven homogenisation erodes signals. | Expect decay of any published calendar/momentum edge; keep monitoring live. |
| Limit-order-book / order-flow | Predictability is "ubiquitous" at high frequency (seconds), driven by order-flow representation more than network depth. | Needs LOB data and HFT-speed execution — unavailable on CFDs and forbidden by FundingPips (HFT / tick scalping). |
| Diffusion models in finance | Mainly synthetic data, scenario / stress generation, denoising; forecasting claims without cost-adjusted OOS evidence. | Useful for risk Monte Carlo, not for direction. |
| Gold regime | Gold decoupled from real yields in 2024–25: central-bank buying (~225 t/quarter 2021–25, 2× 2016–20) is yield-insensitive; record ETF inflows Q3 2025. | The main driver of our long-biased gold legs; a structural regime that can end → keep bear-market protection. |
| Retail "XAUUSD AI" claims (e.g. 85 % win rate, SMC + XGBoost) | Old test windows, no costs / no OOS protocol. | Not credible evidence. |

## Take-aways for the project
1. The 2025–26 evidence agrees with our own: complexity (foundation models, RL, LLM agents, diffusion) does not create
   single-asset directional alpha at retail frequencies after costs.
2. Where the new methods genuinely help: volatility forecasting (TTM + HAR ensemble), synthetic scenarios for risk
   (diffusion / Kronos) — both risk-side, like our cushion rule.
3. Risks to our edge: alpha decay of mechanical signals, and gold's central-bank-driven regime ending.

## Sources
- Pretrained TSFMs for financial return forecasting: https://arxiv.org/abs/2606.27100
- Re(Visiting) TSFMs in finance: https://arxiv.org/html/2511.18578v1
- Kronos: https://arxiv.org/abs/2508.02739
- TSFMs vs econometric volatility benchmarks: https://arxiv.org/abs/2607.05291
- Agent Market Arena (live LLM agents): https://arxiv.org/abs/2510.11695 ; Agentic trading survey: https://arxiv.org/html/2605.19337v1
- BacktestBench: https://arxiv.org/html/2605.17937 ; CLQT: https://arxiv.org/pdf/2606.29771 ; Backtrader-Bench: https://arxiv.org/abs/2608.11232
- DRL critical survey (167 papers): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7363738
- Virtue of complexity rebuttals: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239006 ,
  https://www.su.se/english/divisions/stockholm-business-school/news/articles/2025-09-25-new-research-debunks-the-virtue-of-complexity-in-return-prediction-in-finance ,
  https://bfi.uchicago.edu/wp-content/uploads/2025/08/BFI_WP_2025-104.pdf
- Is trend still your friend?: https://arxiv.org/abs/2607.01550 ; CTA 2025: https://www.toptradersunplugged.com/trend-following-performance-report-august-2025/
- Volatility-managed portfolios critique: https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X
- Alpha decay / crowding: https://arxiv.org/html/2512.11913v1 , https://arxiv.org/html/2605.23905v1
- LOB deep learning guide: https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911
- Diffusion models in finance survey: https://arxiv.org/abs/2608.12583
- Gold decoupling from real yields: https://www.indexbox.io/blog/ftse-russell-rising-yields-pressure-gold-but-central-bank-demand-is-decoupling-the-relationship/ ,
  https://research-center.amundi.com/article/gold-beyond-records

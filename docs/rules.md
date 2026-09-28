# Trading rules — FundingPips 2-Step Standard $100k + user constraints

Sources (checked 2026-09-26; the official help-center pages return 403 to automated fetches, re-verify manually before going live):
- https://help.fundingpips.com/hc/en-us/articles/34501809112081-2-Step-Standard
- https://help.fundingpips.com/hc/en-us/articles/34505029138449-Trading-Conduct-and-Security-Standards
- https://help.fundingpips.com/hc/en-us/articles/34504137479441-News-Trading-Weekend-Holding
- https://proptradingvibes.com/blog/fundingpips-two-step-challenge , https://marketsxplora.com/review/fundingpips-trading-rules/

| Rule | FundingPips 2-Step Standard | Our internal limit (EA hard-coded) |
|---|---|---|
| Profit target | Phase 1: 8 %, Phase 2: 5 % | Once reached, risk off; remaining minimum days with 0.01 lot |
| Daily loss | **3 % or 5 %** (chosen at purchase; 3 % add-on = no minimum trading days) of max(day-open balance, day-open equity), reset 00:00 server (UTC+3), floating included | **3 % hard** (force close), 2 % soft (no new entries); new trade's full risk must fit inside the 3 % room |
| Max loss | 10 % static ($90,000 floor on $100k) | **8 % hard ($92,000)**; cushion rule: risk × (8 − DD)/(8 − 2) above 2 % static DD (min ×0.1) |
| Risk per trade | "Risk Per Trade Idea" rule **removed 2026-07-24** (new purchases); Striking System only in the Monthly 100 % cycle (1 % of account per trade idea) | **≤ 0.5 %** of balance, server-side SL on every order, lot rounded down |
| Minimum days | 3 trading days per phase; no time limit; 30 days inactivity = breach | calendar check |
| Prohibited | HFT, tick scalping, hedging, latency/arbitrage, gap trading, toxic flow, all-or-nothing sizing, server spamming | one position at a time, never opposite positions, no grid/martingale/averaging, ≤ 5 trades/day |
| News | see "News rule" below | **No new position −10…+10 min around any high-impact event of the symbol's currencies (every phase); positions opened < 5 h before the event are closed 10 min before it** |
| Weekend/overnight | Eval: allowed; **Master (Standard): no overnight and no weekend holding** — positions auto-closed every day 20:45–21:00 UTC (summer) / 21:45–22:00 UTC (winter), not a breach, day not counted; Swing add-on allows holding (then FX 1:30, $10/lot) | Overnight allowed Mon–Thu (swaps charged); **flat every Friday before the close** in every phase |

## User trading-style constraints
- "Scalping" = intraday. Each trade targets **≥ $6** move (TP ≥ 600 points).
- Acceptance: Sharpe ≥ 1.5 (daily, √252), daily DD < 3 %, total DD < 8 %, PF ≥ 1.3, ≥ 200 trades, Monte-Carlo 95th pct DD < 8 %.

## Added 2026-09-27
- **Weekly profit target ≥ 2 R** (≥ 1 %/week at 0.5 % risk) — FundingPips payout requires ≥ 2 % profit per payout cycle.
- Daily loss limits are re-based every server day on that day's start reference = max(balance, equity) at 00:00;
  equity charts show them as a per-day floor, not a static line.

## News rule (researched 2026-09-27; official page returns 403 to automated fetch → re-verify on the dashboard)
FundingPips (2-Step Standard, also 1-Step Flex / 2-Step Flex / 2-Step Pro):
- **Evaluation (Phase 1/2):** holding and managing positions through news is allowed; deliberately trading the news
  (opening/closing around the release to exploit it) is prohibited and can close the account.
- **Master (funded):** profits of trades **opened or closed from 5 min before to 5 min after** an event marked
  *Restricted/high impact* in the FundingPips dashboard Economic Calendar are deducted. Speeches: 5 min before start
  to 5 min after the end. If the deduction breaches the daily/max loss, the account is closed. **Exception:** a trade
  opened ≥ 5 hours before the event may be closed inside the window and its profit counts. A partial close flags the
  whole order. Only the affected currency's instruments are restricted.
- FundingPips Zero (not our program): ±10 min, hard breach, no 5-hour exception.
- Calendar = FundingPips dashboard (red/high impact: FOMC, NFP, CPI, ECB, BoE MPC, employment, flash PMIs, …).

Our rule (user instruction 2026-09-27, applies in every phase):
1. **No new position from 10 min before to 10 min after** a high-impact event of the symbol's currencies
   (USD for XAUUSD/NAS100/DJ30; USD + EUR for GER40/EURUSD; USD + JPY for USDJPY). Source: MT5 economic calendar
   "high importance" (broader than FundingPips' red list → conservative).
2. Positions opened **< 5 h before** the event are closed **10 min before** it (so nothing is closed inside the
   ±5 min window). Positions opened ≥ 5 h before may be held (FundingPips exemption).
3. Engine: `symbols.NEWS_BEFORE/AFTER/FLATTEN = 10`, `Guards.news_exempt_min = 290` (age at the flatten bar).

Other FundingPips facts collected: leverage metals 1:30, indices 1:20, forex 1:100 (dynamic leverage on Master for
metals/indices since 2026-03-16); max 20 lots per order; commission $5/lot FX & metals, none on indices; daily loss
reference = max(balance, equity) at 00:00 platform time (UTC+3); consistency 35 % only for the On-Demand payout
cycle (bi-weekly/weekly cycles have none) — we keep 35 % as a design constraint.
Sources: https://propvator.com/blog/funding-pips-news-trading-rule/ ,
https://proptradingvibes.com/blog/fundingpips-rules-overview , https://marketsxplora.com/review/fundingpips-trading-rules/

## FundingPips rule changes 2026 (checked 2026-09-28 on help.fundingpips.com "2 Step Standard" + press)
Removed:
- **SUME** ("Significant Unscheduled Market Event" profit deduction, which had replaced a "Trump post" clause) — removed Aug 2026 after backlash.
- **Risk Per Trade Idea** — removed 24 Jul 2026 06:00 (UTC+3) for 2 Step Standard (new purchases; existing accounts/ZERO keep their terms).
- **Profit Concentration** (one trade idea > 60 % of the target → extra conditions on Master) — removed for new purchases.
- **10 % Phase-1 target option** — no longer offered since 24 Jul 2026.
- Subjective behaviour / "trading flow" reviews — removed 23 Feb 2026.
- Minimum trading days — none when the 3 % daily-loss configuration is bought.
Still in force / added:
- Striking System — only in the new **Monthly 100 %** cycle (purchases ≥ 15 Aug 2026): 1 % loss per trade idea = warning;
  2nd halves the split, 3rd → 20 %, 4th → breach. Monthly also needs 35 % consistency + 7 days ≥ 0.5 % profit.
- 35 % consistency: On Demand (90 %, min 2 %) and Monthly only; Weekly 60 % / Bi-Weekly 80 % have none.
- Master news: ±5 min around high-impact news, speeches 10 min before start → 10 min after end (profit deduction);
  5-hour exemption; evaluation has no news restriction ("purposely trading news" still prohibited).
- Master daily auto-close (above); dynamic leverage on Master metals/indices since 16 Mar 2026 (1:50 → 1:5 above 0.5 lot);
  crypto 1:1 on Master; 20 lots max per click; commission $5/lot FX & metals, 0 on indices.
Impact on v2.2: 11 % of trades are held overnight but they earn ~95 R of 290 R (prefomc 100 % overnight, trendH4,
inside, lw ~30 %) → on a Master account without the Swing add-on these exits are cut at the daily auto-close.

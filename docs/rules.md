# Trading rules — FundingPips 2-Step Standard $10k + user constraints

Sources (checked 2026-09-26; the official help-center pages return 403 to automated fetches, re-verify manually before going live):
- https://help.fundingpips.com/hc/en-us/articles/34501809112081-2-Step-Standard
- https://help.fundingpips.com/hc/en-us/articles/34505029138449-Trading-Conduct-and-Security-Standards
- https://help.fundingpips.com/hc/en-us/articles/34504137479441-News-Trading-Weekend-Holding
- https://proptradingvibes.com/blog/fundingpips-two-step-challenge , https://marketsxplora.com/review/fundingpips-trading-rules/

| Rule | FundingPips 2-Step Standard | Our internal limit (EA hard-coded) |
|---|---|---|
| Profit target | Phase 1: 8 %, Phase 2: 5 % | Once reached, risk off; remaining minimum days with 0.01 lot |
| Daily loss | 5 % of max(day-open balance, day-open equity), reset 00:00 server (UTC+3), floating included | **3 % hard** (force close), 2 % soft (no new entries); new trade's full risk must fit inside the 3 % room |
| Max loss | 10 % static ($9,000 floor) | **8 % hard ($9,200)**, risk halved below 6.5 % DD |
| Risk per trade | trade-idea rules on funded | **≤ 0.5 %** of balance, server-side SL on every order, lot rounded down |
| Minimum days | 3 trading days per phase; no time limit; 30 days inactivity = breach | calendar check |
| Prohibited | HFT, tick scalping, hedging, latency/arbitrage, gap trading, toxic flow, all-or-nothing sizing, server spamming | one position at a time, never opposite positions, no grid/martingale/averaging, ≤ 5 trades/day |
| News | Eval: allowed; Master: ±5 min around high-impact news excluded | **No entries −30/+30 min around USD high-impact news in every phase; flatten ≥ 10 min before** |
| Weekend/overnight | Eval: allowed; Master (Standard): weekend holding not allowed | **Intraday only**: no entries after 22:00 server, flat by 23:45, Friday flat by 22:30 |

## User trading-style constraints
- "Scalping" = intraday. Each trade targets **≥ $6** move (TP ≥ 600 points).
- Acceptance: Sharpe ≥ 1.5 (daily, √252), daily DD < 3 %, total DD < 8 %, PF ≥ 1.3, ≥ 200 trades, Monte-Carlo 95th pct DD < 8 %.

## Added 2026-09-27
- **Weekly profit target ≥ 2 R** (≥ 1 %/week at 0.5 % risk) — FundingPips payout requires ≥ 2 % profit per payout cycle.
- Daily loss limits are re-based every server day on that day's start reference = max(balance, equity) at 00:00;
  equity charts show them as a per-day floor, not a static line.

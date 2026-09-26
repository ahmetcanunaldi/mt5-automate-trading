//+------------------------------------------------------------------+
//| XauPortfolio.mq5                                                 |
//| 9-leg XAUUSD portfolio = research candidate EXP-047              |
//| (D:\mt5-automate-trading\research\final_candidate.py, LEGS9)     |
//|                                                                  |
//| Every leg mirrors its Python definition. Shared rules:           |
//|  - risk per trade = 0.5 % balance (lot rounded down), server SL  |
//|  - max 6 positions, all in ONE direction (no hedging),           |
//|    open risk <= 3 %, one new entry per minute                    |
//|  - daily soft 2 % / hard 3 % (ref = max(bal, eq) at day start),  |
//|    total de-risk 6.5 %, stop 8 %                                 |
//|  - no entries +-30 min around USD high-impact news, every        |
//|    position closed >= 10 min before a release                    |
//|  - everything closed on the last bar before the weekend          |
//+------------------------------------------------------------------+
#property copyright "mt5-automate-trading"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <XauScalper\NewsFilter.mqh>

input double InpInitialBalance = 100000;
input double InpRiskPct        = 0.5;
input double InpDailySoftPct   = 2.0;
input double InpDailyHardPct   = 3.0;
input double InpTotalDeriskPct = 6.5;
input double InpTotalStopPct   = 8.0;
input int    InpMaxPositions   = 6;
input double InpMaxOpenRiskPct = 3.0;
input int    InpMaxTradesDay   = 8;
input double InpCommPerLot     = 7.0;
input int    InpFirstEntryMin  = 65;
input int    InpLastEntryMin   = 1390;
input string InpNewsFile       = "xau_news_server.csv";
input long   InpMagic          = 26092701;
input bool   InpLegTrendH4 = true, InpLegTday = true, InpLegDrift = true, InpLegFriday = true, InpLegTom = true;
input bool   InpLegLW = true, InpLegInside = true, InpLegNR7 = true, InpLegFriClose = true;

CTrade      g_trade;
CNewsFilter g_news;

struct SPos
  {
   ulong    ticket;
   datetime entry;
   int      hold_min;
   double   trail;
   double   risk_usd;
   string   leg;
  };
SPos     g_pos[];
datetime g_day = 0, g_last_m1 = 0, g_last_m15 = 0, g_last_h4 = 0, g_last_entry_min = 0;
double   g_day_ref = 0;
int      g_day_trades = 0;
bool     g_stopped = false;
// once-per-day leg flags
bool     g_lw_done, g_inside_done, g_nr7_done, g_tday_done, g_drift_done, g_friday_done, g_tom_done, g_fric_done;

//--------------------------------------------------------------- helpers
int    MinuteOfDay(datetime t) { return (int)((t % 86400) / 60); }
int    Dow(datetime t)         { return (int)((t / 86400 + 4) % 7); }     // 0 = Sunday
datetime DayOf(datetime t)     { return (datetime)(t / 86400 * 86400); }

double WilderATR(ENUM_TIMEFRAMES tf, int period, int shift, int lookback = 400)
  {
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int n = CopyRates(_Symbol, tf, shift, lookback, r);
   if(n < period + 2) return 0.0;
   double a = 0;
   for(int i = 0; i < n; i++)
     {
      double tr = r[i].high - r[i].low;
      if(i > 0) tr = MathMax(tr, MathMax(MathAbs(r[i].high - r[i - 1].close), MathAbs(r[i].low - r[i - 1].close)));
      a = (i == 0) ? tr : a + (tr - a) / period;
     }
   return a;
  }

double EMA(const MqlRates &r[], int last, int span)
  {
   double al = 2.0 / (span + 1.0), e = 0;
   for(int i = 0; i <= last; i++) e = (i == 0) ? r[i].close : e + al * (r[i].close - e);
   return e;
  }

// D1 trend state of the PREVIOUS day: +1 (close > EMA50 and EMA20 > EMA50), -1 mirror, 0 otherwise
int DailyState()
  {
   MqlRates d[];
   ArraySetAsSeries(d, false);
   int n = CopyRates(_Symbol, PERIOD_D1, 1, 500, d);
   if(n < 100) return 0;
   double e20 = EMA(d, n - 1, 20), e50 = EMA(d, n - 1, 50), c = d[n - 1].close;
   if(c > e50 && e20 > e50) return 1;
   if(c < e50 && e20 < e50) return -1;
   return 0;
  }

int CountOpen(int &dir, double &open_risk)
  {
   dir = 0; open_risk = 0;
   int k = 0;
   for(int i = ArraySize(g_pos) - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(g_pos[i].ticket)) { ArrayRemove(g_pos, i, 1); continue; }
      dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
      open_risk += g_pos[i].risk_usd;
      k++;
     }
   return k;
  }

void CloseTicket(int i, string why)
  {
   if(g_trade.PositionClose(g_pos[i].ticket))
      PrintFormat("[Exit] %s %s", g_pos[i].leg, why);
  }

void CloseAll(string why)
  {
   for(int i = ArraySize(g_pos) - 1; i >= 0; i--) CloseTicket(i, why);
  }

//--------------------------------------------------------------- entry
bool Enter(int dir, double sl_dist, double tp_dist, int hold_min, double trail, string leg)
  {
   datetime now = TimeCurrent();
   int tm = MinuteOfDay(now);
   if(now / 60 == g_last_entry_min) return false;                     // one entry per minute (engine parity)
   if(g_stopped || tm < InpFirstEntryMin || tm > InpLastEntryMin) return false;
   if(g_news.BlockEntry(now) || g_day_trades >= InpMaxTradesDay) return false;
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double day_pnl = bal - g_day_ref;
   if(day_pnl <= -InpDailySoftPct / 100.0 * g_day_ref) return false;
   int cur_dir; double open_risk;
   int nopen = CountOpen(cur_dir, open_risk);
   if(nopen >= InpMaxPositions || (nopen > 0 && cur_dir != dir)) return false;         // no hedging
   if(sl_dist <= 0) return false;
   double rp = ((InpInitialBalance - bal) / InpInitialBalance * 100.0 < InpTotalDeriskPct) ? InpRiskPct : InpRiskPct * 0.5;
   double room = InpDailyHardPct / 100.0 * g_day_ref + day_pnl - open_risk;
   double risk = MathMin(MathMin(rp / 100.0 * bal, room), InpMaxOpenRiskPct / 100.0 * bal - open_risk);
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS), ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP), vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double per_lot = sl_dist / ts * tv + InpCommPerLot;
   double lots = MathFloor(risk / per_lot / step + 1e-9) * step;
   if(lots < vmin) return false;
   int dg = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   bool ok;
   if(dir > 0)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      ok = g_trade.Buy(lots, _Symbol, 0, NormalizeDouble(ask - sl_dist, dg), tp_dist > 0 ? NormalizeDouble(ask + tp_dist, dg) : 0, leg);
     }
   else
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      ok = g_trade.Sell(lots, _Symbol, 0, NormalizeDouble(bid + sl_dist, dg), tp_dist > 0 ? NormalizeDouble(bid - tp_dist, dg) : 0, leg);
     }
   if(!ok || (g_trade.ResultRetcode() != TRADE_RETCODE_DONE && g_trade.ResultRetcode() != TRADE_RETCODE_PLACED)) return false;
   SPos p;
   p.ticket = g_trade.ResultOrder(); p.entry = now; p.hold_min = hold_min; p.trail = trail;
   p.risk_usd = lots * per_lot; p.leg = leg;
   // resolve position ticket (hedging account: position id == order ticket in most brokers; fall back to scan)
   if(!PositionSelectByTicket(p.ticket))
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong t = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) == InpMagic && PositionGetString(POSITION_COMMENT) == leg) { p.ticket = t; break; }
        }
   int k = ArraySize(g_pos);
   ArrayResize(g_pos, k + 1);
   g_pos[k] = p;
   g_day_trades++;
   g_last_entry_min = now / 60;
   PrintFormat("[Entry] %s %s %.2f lots sl=%.2f hold=%d trail=%.2f", leg, dir > 0 ? "BUY" : "SELL", lots, sl_dist, hold_min, trail);
   return true;
  }

//--------------------------------------------------------------- legs
void LegsOnM1(datetime now)
  {
   int tm = MinuteOfDay(now), dw = Dow(now);
   double atr_d = WilderATR(PERIOD_D1, 14, 1);
   // drift: Tue-Fri 01:15 long, SL 4 ATR_H1, hold 475 min
   if(InpLegDrift && !g_drift_done && tm >= 75 && dw >= 2 && dw <= 5)
     { g_drift_done = true; Enter(1, 4.0 * WilderATR(PERIOD_H1, 14, 1), 0, 475, 0, "drift"); }
   // friday: Friday 01:05 long, SL 1.5 ATR_D, hold 1315 min
   if(InpLegFriday && !g_friday_done && tm >= 65 && dw == 5)
     { g_friday_done = true; Enter(1, 1.5 * atr_d, 0, 1315, 0, "friday"); }
   // tom: first trading day of the month 01:10 long, SL 2 ATR_D, hold 3 days - 60 min
   if(InpLegTom && !g_tom_done && tm >= 70)
     {
      g_tom_done = true;
      MqlDateTime a, b;
      TimeToStruct(now, a);
      TimeToStruct(iTime(_Symbol, PERIOD_D1, 1), b);
      if(a.mon != b.mon) Enter(1, 2.0 * atr_d, 0, 3 * 1440 - 60, 0, "tom");
     }
   // fri_close: Friday 23:05 long, SL 4 ATR_M15, hold 50 min
   if(InpLegFriClose && !g_fric_done && tm >= 1385 && dw == 5)
     { g_fric_done = true; Enter(1, 4.0 * WilderATR(PERIOD_M15, 14, 1), 0, 50, 0, "fri_close"); }
  }

void LegsOnM15(datetime now)
  {
   MqlRates b[];
   ArraySetAsSeries(b, false);
   datetime day = DayOf(now);
   int n = CopyRates(_Symbol, PERIOD_M15, day, now - 1, b);          // today's CLOSED M15 bars
   if(n < 1) return;
   MqlRates last = b[n - 1];
   int tm_open = MinuteOfDay(last.time);
   double o = b[0].open, hi = b[0].high, lo = b[0].low;
   for(int i = 0; i < n; i++) { hi = MathMax(hi, b[i].high); lo = MathMin(lo, b[i].low); }
   double atr_d = WilderATR(PERIOD_D1, 14, 1);
   int st = DailyState();
   MqlRates d[];
   ArraySetAsSeries(d, true);
   if(CopyRates(_Symbol, PERIOD_D1, 1, 8, d) < 8) return;              // d[0] = yesterday
   double pdh = d[0].high, pdl = d[0].low, prev_rng = d[0].high - d[0].low;
   // tday: at the M15 bar opening 18:00, move >= 0.3 ATR_D with the D1 state, close in top/bottom 25 %
   if(InpLegTday && !g_tday_done && tm_open == 1080 && st != 0 && atr_d > 0)
     {
      g_tday_done = true;
      double pos = (hi > lo) ? (last.close - lo) / (hi - lo) : 0.5;
      if((last.close - o) * st >= 0.3 * atr_d && ((st == 1 && pos >= 0.75) || (st == -1 && pos <= 0.25)))
         Enter(st, 1.0 * atr_d, 0, 330, 0, "tday");
     }
   // lw: from 10:00 to 20:00, first close beyond open +/- 0.4 prev range, SL 1 ATR_D, hold 1440
   if(InpLegLW && !g_lw_done && tm_open >= 600 && tm_open <= 1200 && prev_rng > 0)
     {
      int dir = 0;
      if(last.close > o + 0.4 * prev_rng) dir = 1; else if(last.close < o - 0.4 * prev_rng) dir = -1;
      if(dir != 0) { g_lw_done = true; Enter(dir, 1.0 * atr_d, 0, 1440, 0, "lw"); }
     }
   // inside / nr7 day breakouts with the D1 state, 01:05..20:00
   bool inside = (d[0].high < d[1].high && d[0].low > d[1].low);
   double mn = DBL_MAX;
   for(int i = 0; i < 7; i++) mn = MathMin(mn, d[i].high - d[i].low);
   bool nr7 = (prev_rng <= mn + 1e-9);
   if(tm_open >= 65 && tm_open <= 1200 && st != 0)
     {
      int dir = 0;
      if(last.close > pdh) dir = 1; else if(last.close < pdl) dir = -1;
      if(dir != 0)
        {
         double risk = MathMin(MathAbs(last.close - (dir == 1 ? pdl : pdh)), 1.5 * atr_d);
         if(InpLegInside && inside && !g_inside_done) { g_inside_done = true; if(dir == st) Enter(dir, risk, 0, 1440, 0, "inside"); }
         if(InpLegNR7 && nr7 && !g_nr7_done) { g_nr7_done = true; if(dir == st) Enter(dir, risk, 0, 1440, 0, "nr7"); }
        }
     }
  }

void LegsOnH4()
  {
   if(!InpLegTrendH4) return;
   MqlRates h[];
   ArraySetAsSeries(h, false);
   int n = CopyRates(_Symbol, PERIOD_H4, 1, 400, h);               // closed H4 bars, last = h[n-1]
   if(n < 200) return;
   double hi = -DBL_MAX;
   for(int i = n - 181; i < n - 1; i++) hi = MathMax(hi, h[i].high);  // previous 180 bars
   if(h[n - 1].close > hi)
     {
      double a = 0;                                                   // Wilder ATR(20) on H4
      for(int i = 0; i < n; i++)
        {
         double tr = h[i].high - h[i].low;
         if(i > 0) tr = MathMax(tr, MathMax(MathAbs(h[i].high - h[i - 1].close), MathAbs(h[i].low - h[i - 1].close)));
         a = (i == 0) ? tr : a + (tr - a) / 20.0;
        }
      Enter(1, 2.0 * a, 0, 240 * 240, 6.0 * a, "trendH4");
     }
  }

//--------------------------------------------------------------- management
void Manage(datetime now, bool new_m1)
  {
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq - g_day_ref <= -InpDailyHardPct / 100.0 * g_day_ref) { CloseAll("daily hard guard"); return; }
   if(g_news.MustFlatten(now)) { CloseAll("news flatten"); return; }
   // weekend: last minutes of Friday
   if(Dow(now) == 5 && MinuteOfDay(now) >= 1435) { CloseAll("weekend"); return; }
   int dg = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   for(int i = ArraySize(g_pos) - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(g_pos[i].ticket)) { ArrayRemove(g_pos, i, 1); continue; }
      if(now >= g_pos[i].entry + g_pos[i].hold_min * 60) { CloseTicket(i, "max hold"); continue; }
      if(new_m1 && g_pos[i].trail > 0)
        {
         double sl = PositionGetDouble(POSITION_SL), tp = PositionGetDouble(POSITION_TP);
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
           {
            double nsl = NormalizeDouble(iHigh(_Symbol, PERIOD_M1, 1) - g_pos[i].trail, dg);
            if(nsl > sl + _Point && nsl < SymbolInfoDouble(_Symbol, SYMBOL_BID)) g_trade.PositionModify(g_pos[i].ticket, nsl, tp);
           }
         else
           {
            double spr = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double nsl = NormalizeDouble(iLow(_Symbol, PERIOD_M1, 1) + spr + g_pos[i].trail, dg);
            if(nsl < sl - _Point && nsl > SymbolInfoDouble(_Symbol, SYMBOL_ASK)) g_trade.PositionModify(g_pos[i].ticket, nsl, tp);
           }
        }
     }
  }

//--------------------------------------------------------------- events
int OnInit()
  {
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(30);
   if(!g_news.Load(InpNewsFile, 30, 30, 10)) { Print("[Init] news file missing - refusing to trade"); return INIT_FAILED; }
   return INIT_SUCCEEDED;
  }

void OnTick()
  {
   datetime now = TimeCurrent();
   datetime day = DayOf(now);
   if(day != g_day)
     {
      g_day = day;
      g_day_ref = MathMax(AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));
      g_day_trades = 0;
      g_lw_done = g_inside_done = g_nr7_done = g_tday_done = g_drift_done = g_friday_done = g_tom_done = g_fric_done = false;
     }
   if(!g_stopped && (InpInitialBalance - AccountInfoDouble(ACCOUNT_EQUITY)) / InpInitialBalance * 100.0 >= InpTotalStopPct)
     {
      g_stopped = true; CloseAll("total stop");
      Print("[Guard] TOTAL STOP - trading disabled");
     }
   datetime m1 = iTime(_Symbol, PERIOD_M1, 0);
   bool new_m1 = (m1 != g_last_m1);
   Manage(now, new_m1);
   if(!new_m1 || g_stopped) return;
   g_last_m1 = m1;
   datetime h4 = iTime(_Symbol, PERIOD_H4, 0);
   if(h4 != g_last_h4) { g_last_h4 = h4; LegsOnH4(); }
   datetime m15 = iTime(_Symbol, PERIOD_M15, 0);
   if(m15 != g_last_m15) { g_last_m15 = m15; LegsOnM15(now); }
   LegsOnM1(now);
  }

//+------------------------------------------------------------------+
//| XauScalper.mq5                                                   |
//| XAUUSD intraday breakout portfolio for FundingPips 2-Step        |
//| (research: D:\mt5-automate-trading, EXP-005 central parameters)  |
//|                                                                  |
//| Hard rules: 1 position at a time (no hedge, no grid/martingale), |
//| server-side SL on every order, risk <= InpRiskPct of balance,    |
//| TP >= $6, no entries around USD high-impact news, flat by day end|
//+------------------------------------------------------------------+
#property copyright "mt5-automate-trading"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <XauScalper\PropGuard.mqh>
#include <XauScalper\NewsFilter.mqh>
#include <XauScalper\Signals.mqh>

input group "Account / FundingPips"
input double InpInitialBalance = 10000;   // challenge initial balance
input double InpTargetPct      = 8.0;     // phase target % (P1 8, P2 5); 0 = no lock
input int    InpMinTradingDays = 3;       // FP minimum trading days per phase
input double InpRiskPct        = 0.5;     // risk per trade, % of balance
input double InpDailySoftPct   = 2.0;     // no new entries below this day loss %
input double InpDailyHardPct   = 3.0;     // close all at this day loss % (incl. floating)
input double InpTotalDeriskPct = 6.5;     // halve risk below this total DD %
input double InpTotalStopPct   = 8.0;     // stop trading at this total DD %
input double InpCommPerLot     = 7.0;     // round-turn commission per lot (sizing only)
input int    InpMaxTradesDay   = 5;
input int    InpMaxSpreadPts   = 80;      // skip entries when spread is wider

input group "Session (server time, minutes of day)"
input int    InpFirstEntryMin  = 65;      // 01:05
input int    InpLastEntryMin   = 1320;    // 22:00
input int    InpFlattenMin     = 1425;    // 23:45
input int    InpFriFlattenMin  = 1350;    // Friday 22:30

input group "News filter"
input string InpNewsFile       = "xau_news_server.csv";
input int    InpNewsBeforeMin  = 30;
input int    InpNewsAfterMin   = 30;
input int    InpNewsFlatMin    = 10;

input group "Leg 1: NY opening-range breakout"
input bool   InpOrbOn        = true;
input int    InpOrbStart     = 990;       // 16:30
input int    InpOrbBars      = 2;
input int    InpOrbEnd       = 1200;
input double InpOrbSlAtr     = 2.0;
input double InpOrbRR        = 2.0;
input double InpOrbTrailAtr  = 0.0;
input double InpOrbMinRngAtr = 0.5;
input double InpOrbMaxRngAtr = 5.0;
input int    InpOrbHold      = 480;
input int    InpOrbTrend     = 1;         // 0 none, 1 with H1 trend, 2 long only

input group "Leg 2: Donchian breakout (US session)"
input bool   InpDonOn        = true;
input int    InpDonN         = 32;
input int    InpDonStart     = 900;       // 15:00
input int    InpDonEnd       = 1320;
input int    InpDonPerDay    = 2;
input double InpDonSlAtr     = 2.0;
input double InpDonRR        = 3.0;
input double InpDonTrailAtr  = 0.0;
input double InpDonMinWidth  = 4.0;
input int    InpDonHold      = 480;
input int    InpDonTrend     = 1;

input group "Leg 3: previous-day high/low break"
input bool   InpPdbOn        = true;
input int    InpPdbStart     = 120;
input int    InpPdbEnd       = 900;
input double InpPdbSlAtr     = 2.0;
input double InpPdbRR        = 3.0;
input double InpPdbTrailAtr  = 2.0;
input double InpPdbBuffer    = 0.3;
input int    InpPdbHold      = 60;
input int    InpPdbTrend     = 2;

input group "Misc"
input long   InpMagic        = 26092601;

CTrade       g_trade;
CPropGuard   g_guard;
CNewsFilter  g_news;
CSignals     g_sig;
SLegOrb      g_orb;
SLegDonchian g_don;
SLegPdb      g_pdb;

datetime g_last_m15 = 0;
datetime g_last_m1 = 0;
datetime g_entry_time = 0;
int      g_hold_min = 0;
double   g_trail = 0;
string   g_leg = "";
bool     g_target_locked = false;

//+------------------------------------------------------------------+
int MinuteOfDay(datetime t) { return (int)((t % 86400) / 60); }

bool GetPosition(ulong &ticket)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
        {
         ticket = t;
         return true;
        }
     }
   return false;
  }

void CloseAll(string why)
  {
   ulong t;
   while(GetPosition(t))
     {
      if(!g_trade.PositionClose(t))
        {
         PrintFormat("[Exit] close failed %d", GetLastError());
         return;
        }
      PrintFormat("[Exit] %s", why);
     }
  }

int TradingDaysSinceStart()
  {
   // distinct server days with an entry deal by this EA
   if(!HistorySelect(0, TimeCurrent()))
      return 0;
   int n = 0;
   datetime last = 0;
   for(int i = 0; i < HistoryDealsTotal(); i++)
     {
      ulong d = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(d, DEAL_MAGIC) != InpMagic || HistoryDealGetInteger(d, DEAL_ENTRY) != DEAL_ENTRY_IN)
         continue;
      datetime day = (datetime)(HistoryDealGetInteger(d, DEAL_TIME) / 86400 * 86400);
      if(day != last) { n++; last = day; }
     }
   return n;
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   if(_Symbol != "XAUUSD" && StringFind(_Symbol, "XAU") < 0)
      Print("[Init] WARNING: designed for XAUUSD");
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(30);
   g_guard.Init(InpInitialBalance, InpRiskPct, InpDailySoftPct, InpDailyHardPct, InpTotalDeriskPct,
                InpTotalStopPct, InpCommPerLot, InpMaxTradesDay);
   if(!g_news.Load(InpNewsFile, InpNewsBeforeMin, InpNewsAfterMin, InpNewsFlatMin))
     {
      Print("[Init] news file missing - refusing to trade (safety)");
      return INIT_FAILED;
     }
   g_sig.Init(_Symbol);
   g_orb.on = InpOrbOn; g_orb.or_start = InpOrbStart; g_orb.or_bars = InpOrbBars; g_orb.end_min = InpOrbEnd;
   g_orb.sl_atr = InpOrbSlAtr; g_orb.rr = InpOrbRR; g_orb.trail_atr = InpOrbTrailAtr;
   g_orb.min_rng_atr = InpOrbMinRngAtr; g_orb.max_rng_atr = InpOrbMaxRngAtr; g_orb.hold_min = InpOrbHold;
   g_orb.trend = InpOrbTrend;
   g_don.on = InpDonOn; g_don.n = InpDonN; g_don.start_min = InpDonStart; g_don.end_min = InpDonEnd;
   g_don.per_day = InpDonPerDay; g_don.sl_atr = InpDonSlAtr; g_don.rr = InpDonRR; g_don.trail_atr = InpDonTrailAtr;
   g_don.min_width_atr = InpDonMinWidth; g_don.hold_min = InpDonHold; g_don.trend = InpDonTrend;
   g_pdb.on = InpPdbOn; g_pdb.start_min = InpPdbStart; g_pdb.end_min = InpPdbEnd; g_pdb.sl_atr = InpPdbSlAtr;
   g_pdb.rr = InpPdbRR; g_pdb.trail_atr = InpPdbTrailAtr; g_pdb.buffer_atr = InpPdbBuffer;
   g_pdb.hold_min = InpPdbHold; g_pdb.trend = InpPdbTrend;
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void ManagePosition(datetime now)
  {
   ulong t;
   if(!GetPosition(t))
      return;
   int tm = MinuteOfDay(now);
   int dow = (int)((now / 86400 + 4) % 7);            // 0 = Sunday
   if(g_guard.HardBreach())                             { CloseAll("daily hard guard"); return; }
   if(g_news.MustFlatten(now))                          { CloseAll("news flatten"); return; }
   if(tm >= InpFlattenMin || (dow == 5 && tm >= InpFriFlattenMin)) { CloseAll("end of day"); return; }
   if(g_entry_time > 0 && now >= g_entry_time + g_hold_min * 60) { CloseAll("max hold"); return; }
   // trailing stop on completed M1 bars (one modification per minute at most)
   if(g_trail > 0)
     {
      datetime m1 = iTime(_Symbol, PERIOD_M1, 0);
      if(m1 != g_last_m1)
        {
         g_last_m1 = m1;
         if(!PositionSelectByTicket(t))
            return;
         long type = PositionGetInteger(POSITION_TYPE);
         double sl = PositionGetDouble(POSITION_SL), tp = PositionGetDouble(POSITION_TP);
         double ask_spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
         int dg = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
         if(type == POSITION_TYPE_BUY)
           {
            double nsl = NormalizeDouble(iHigh(_Symbol, PERIOD_M1, 1) - g_trail, dg);
            if(nsl > sl + _Point && nsl < SymbolInfoDouble(_Symbol, SYMBOL_BID))
               g_trade.PositionModify(t, nsl, tp);
           }
         else
           {
            double nsl = NormalizeDouble(iLow(_Symbol, PERIOD_M1, 1) + ask_spread + g_trail, dg);
            if(nsl < sl - _Point && nsl > SymbolInfoDouble(_Symbol, SYMBOL_ASK))
               g_trade.PositionModify(t, nsl, tp);
           }
        }
     }
  }

//+------------------------------------------------------------------+
void TryEntry(datetime now)
  {
   datetime m15 = iTime(_Symbol, PERIOD_M15, 0);
   if(m15 == g_last_m15)
      return;
   g_last_m15 = m15;
   ulong t;
   if(GetPosition(t))
      return;                                           // one position at a time
   SSignal s;
   if(!g_sig.Evaluate(g_orb, g_don, g_pdb, s))
      return;
   string why = "";
   int tm = MinuteOfDay(now);
   int dow = (int)((now / 86400 + 4) % 7);
   bool min_lot_only = false;
   if(InpTargetPct > 0)
     {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(eq >= InpInitialBalance * (1 + InpTargetPct / 100.0))
        {
         int days = TradingDaysSinceStart();
         if(days >= InpMinTradingDays)
           {
            if(!g_target_locked) PrintFormat("[Target] reached with %d trading days - trading locked", days);
            g_target_locked = true;
            return;
           }
         min_lot_only = true;                           // just collect the remaining minimum days
        }
     }
   if(tm < InpFirstEntryMin || tm > InpLastEntryMin)        why = "session";
   else if(dow == 5 && tm >= InpFriFlattenMin - 60)         why = "friday";
   else if(g_news.BlockEntry(now))                          why = "news";
   else if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > InpMaxSpreadPts) why = "spread";
   else if(!g_guard.EntryAllowed(why))                      {}
   if(why != "")
     {
      PrintFormat("[Skip] %s %s: %s", s.leg, s.dir > 0 ? "BUY" : "SELL", why);
      return;
     }
   double lots = g_guard.Lots(_Symbol, s.sl, min_lot_only);
   if(lots <= 0)
     {
      PrintFormat("[Skip] %s lots=0 (sl=%.2f)", s.leg, s.sl);
      return;
     }
   int dg = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   bool ok;
   if(s.dir > 0)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      ok = g_trade.Buy(lots, _Symbol, 0, NormalizeDouble(ask - s.sl, dg), NormalizeDouble(ask + s.tp, dg), s.leg);
     }
   else
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      ok = g_trade.Sell(lots, _Symbol, 0, NormalizeDouble(bid + s.sl, dg), NormalizeDouble(bid - s.tp, dg), s.leg);
     }
   if(ok && (g_trade.ResultRetcode() == TRADE_RETCODE_DONE || g_trade.ResultRetcode() == TRADE_RETCODE_PLACED))
     {
      g_guard.CountTrade();
      g_entry_time = now; g_hold_min = s.hold_min; g_trail = s.trail; g_leg = s.leg;
      PrintFormat("[Entry] %s %s %.2f lots sl=%.2f tp=%.2f atr=%.2f h1(e50=%.2f e200=%.2f c=%.2f)",
                  s.leg, s.dir > 0 ? "BUY" : "SELL", lots, s.sl, s.tp, g_sig.atr, g_sig.h1_e50, g_sig.h1_e200,
                  g_sig.h1_close);
     }
   else
      PrintFormat("[Entry] FAILED %s ret=%d", s.leg, g_trade.ResultRetcode());
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   datetime now = TimeCurrent();
   g_guard.OnTickUpdate();
   ManagePosition(now);
   if(g_target_locked || g_guard.Stopped())
      return;
   TryEntry(now);
  }

//+------------------------------------------------------------------+
double OnTester()
  {
   // custom criterion: daily-return Sharpe is computed in Python from the report; return recovery factor
   return TesterStatistics(STAT_RECOVERY_FACTOR);
  }

//+------------------------------------------------------------------+
//| XauIdxPortfolio.mq5                                              |
//| Algorithm v2 (docs/ALGORITHM.md, research/best_v2.py, EXP-087):  |
//|   XAUUSD 12 legs (best13) + NAS100 / DJ30 6 legs each,           |
//|   one account, shared guards.                                    |
//| Attach to an XAUUSD chart (any timeframe). Index symbols are     |
//| traded from the same EA (multi-symbol).                          |
//|                                                                  |
//| Shared rules (mirror research/engine_multi.py):                  |
//|  - risk 0.5 % of balance (or of InpInitialBalance when           |
//|    InpRiskOnInitial), lot rounded down to the symbol step        |
//|  - max 6 positions in total, one direction PER SYMBOL, open      |
//|    risk <= 3 %, max 8 entries per server day                     |
//|  - one signal per symbol per minute (first leg in book order)    |
//|  - daily soft 2 % / hard 3 % (ref = max(bal, eq) at day start),  |
//|    total stop 8 %; risk scales down linearly from 1 at a static  |
//|    DD of InpCushionStartPct (2 %) to 0.1 at 8 % (cushion rule,   |
//|    QM-008/009, lockbox-validated; 0 = old step rule at 6.5 %)    |
//|  - news v2: no entries -10..+10 min around high-impact news,     |
//|    positions closed 10 min before                                |
//|  - everything closed before the weekend                          |
//|  - funded mode: day profit cap (close all, stop for the day)     |
//| v2.21 live readiness (no effect in the tester):                  |
//|  - server time zone check (must be NY close = GMT+2/+3, US DST)  |
//|  - state file: open positions (hold / trail / leg), day ref,     |
//|    day counters, once-per-day flags, stop / cap survive restarts |
//|  - entries appended to the CSV at entry time, deals CSV rewritten|
//|    every new server day; news / FOMC file coverage checks        |
//|  - order filling mode taken from the symbol                      |
//| v2.22 (live only): demo-only guard; symbols with trading disabled|
//|  refuse to start; broker session end respected: no entries in    |
//|  the last 15 min, same-day positions and the Friday book closed  |
//|  5 min before the symbol's trade session ends                    |
//| v2.24: lot sizing from OrderCalcProfit (the server's own P&L     |
//|  formula) - MetaQuotes-Demo reports XAUUSD tick value 0.10 for   |
//|  a 100 oz contract (true 1.00), which made v2.23 size 10x (5 %)  |
//+------------------------------------------------------------------+
#property copyright "mt5-automate-trading"
#property version   "2.24"

#include <Trade\Trade.mqh>
#include <XauScalper\NewsFilter.mqh>

input double InpInitialBalance = 100000;
input double InpRiskPct        = 0.5;
input bool   InpRiskOnInitial  = false;     // funded mode: size from the initial balance
input double InpDayProfitCap   = 0.0;       // funded mode: 1.25 (% of day reference), 0 = off
input double InpDailySoftPct   = 2.0;
input double InpDailyHardPct   = 3.0;
input double InpTotalDeriskPct = 6.5;       // used only when InpCushionStartPct = 0
input double InpCushionStartPct = 2.0;      // cushion rule start (static DD %), 0 = step rule
input double InpTotalStopPct   = 8.0;
input int    InpMaxPositions   = 6;
input double InpMaxOpenRiskPct = 3.0;
input int    InpMaxTradesDay   = 8;
input double InpCommXau        = 7.0;       // $/lot used for sizing only
input double InpCommIdx        = 0.0;
input int    InpFirstEntryMin  = 65;
input int    InpLastEntryMin   = 1390;
input int    InpNewsBefore     = 10;
input int    InpNewsAfter      = 10;
input int    InpNewsFlatten    = 10;
input string InpNewsFile       = "xau_news_server.csv";
input string InpFomcFile       = "fomc_server.csv";
input string InpNasSymbol      = "NAS100.r";
input string InpDjSymbol       = "DJ30.r";
input datetime InpDriftMedStart = D'2017.12.29';   // start of the expanding median (research data start)
input long   InpMagic          = 27092602;
input bool   InpTradeXau = true, InpTradeNas = true, InpTradeDj = true;
input bool   InpExportTrades = true;        // write <prefix>_entries.csv / <prefix>_deals.csv to Common\Files at the end
input string InpExportPrefix = "xauidx";
input bool   InpCheckServerTime = true;     // live: refuse to trade unless server time = NY + 7 h (research convention)
input bool   InpDemoOnly = true;            // live: refuse to run on anything but a demo account

CTrade      g_trade;
CNewsFilter g_news;

#define NSYM 3
string   g_sym[NSYM];
bool     g_on[NSYM];
double   g_comm[NSYM];
datetime g_slot[NSYM];             // minute already used by a signal of this symbol

struct SPos
  {
   ulong    ticket;
   int      s;
   datetime entry;
   int      hold_min;
   double   trail;
   double   risk_usd;
   string   leg;
  };
SPos     g_pos[];
string   g_entry_log[];                     // one CSV line per opened position (export)
datetime g_fomc[];
datetime g_day = 0, g_last_m1 = 0, g_last_m15 = 0, g_last_h4 = 0;
double   g_day_ref = 0;
int      g_day_trades = 0;
bool     g_stopped = false, g_capped = false;
bool     g_drift_low = false;
// once-per-day flags
bool     g_lw_done, g_inside_done, g_nr7_done, g_tday_done, g_tday900_done, g_drift_done, g_friday_done, g_tom_done,
         g_fric_done, g_sc_done, g_season_done;
bool     g_idx_done[NSYM];
bool     g_fomc_done[NSYM];
bool     g_live = false;                    // not in the strategy tester

//--------------------------------------------------------------- live helpers (v2.21)
string StateFile() { return InpExportPrefix + "_state.csv"; }

// account-currency loss of 1.0 lot if the stop (sl_dist away from the current entry side) is hit; uses the server's
// own profit formula (OrderCalcProfit) and falls back to tick value / tick size only if that fails
double LossPerLot(string sym, int dir, double sl_dist)
  {
   double px = dir > 0 ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID), pl = 0.0;
   if(px > 0 && OrderCalcProfit(dir > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, sym, 1.0, px, dir > 0 ? px - sl_dist : px + sl_dist, pl) && pl < 0)
      return -pl;
   double tv = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_LOSS), ts = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   return (tv > 0 && ts > 0) ? sl_dist / ts * tv : 0.0;
  }

// end of the symbol's last trade session of the day containing t (0 = unknown / open until midnight)
datetime SessionEnd(string sym, datetime t)
  {
   MqlDateTime d; TimeToStruct(t, d);
   datetime from, to, best = 0;
   for(uint i = 0; i < 10; i++)
     {
      if(!SymbolInfoSessionTrade(sym, (ENUM_DAY_OF_WEEK)d.day_of_week, i, from, to)) break;
      if(to > best) best = to;
     }
   if(best == 0 || best >= 86400 - 60) return 0;
   return DayOf(t) + best;
  }

// research convention: server time = New York time + 7 h (GMT+2 winter / GMT+3 US summer time)
int ExpectedServerOffset(datetime gmt)
  {
   MqlDateTime d; TimeToStruct(gmt, d);
   MqlDateTime a; a.year = d.year; a.mon = 3; a.day = 1; a.hour = 7; a.min = 0; a.sec = 0;   // 2 am EST = 07:00 GMT
   datetime mar1 = StructToTime(a);
   int dw = (int)((mar1 / 86400 + 4) % 7);
   datetime start = mar1 + ((7 - dw) % 7 + 7) * 86400;                                    // second Sunday of March
   MqlDateTime b; b.year = d.year; b.mon = 11; b.day = 1; b.hour = 6; b.min = 0; b.sec = 0;  // 2 am EDT = 06:00 GMT
   datetime nov1 = StructToTime(b);
   dw = (int)((nov1 / 86400 + 4) % 7);
   datetime end = nov1 + ((7 - dw) % 7) * 86400;                                           // first Sunday of November
   return (gmt >= start && gmt < end) ? 3 * 3600 : 2 * 3600;
  }

void AppendEntry(string line)
  {
   string f = InpExportPrefix + "_entries.csv";
   bool fresh = !FileIsExist(f, FILE_COMMON);
   int h = FileOpen(f, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return;
   if(fresh) FileWriteString(h, "position,symbol,leg,dir,time,price_at_signal,sl_price,sl_dist,lots,risk_usd,balance,risk_pct,risk_mult_used\n");
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, line + "\n");
   FileClose(h);
  }

int Flags()
  {
   bool f[11];
   f[0] = g_lw_done; f[1] = g_inside_done; f[2] = g_nr7_done; f[3] = g_tday_done; f[4] = g_tday900_done;
   f[5] = g_drift_done; f[6] = g_friday_done; f[7] = g_tom_done; f[8] = g_fric_done; f[9] = g_sc_done; f[10] = g_season_done;
   int m = 0;
   for(int i = 0; i < 11; i++) if(f[i]) m |= (1 << i);
   for(int s = 0; s < NSYM; s++) { if(g_idx_done[s]) m |= (1 << (11 + s)); if(g_fomc_done[s]) m |= (1 << (14 + s)); }
   return m;
  }

void SetFlags(int m)
  {
   g_lw_done = (m & 1) != 0; g_inside_done = (m & 2) != 0; g_nr7_done = (m & 4) != 0; g_tday_done = (m & 8) != 0;
   g_tday900_done = (m & 16) != 0; g_drift_done = (m & 32) != 0; g_friday_done = (m & 64) != 0; g_tom_done = (m & 128) != 0;
   g_fric_done = (m & 256) != 0; g_sc_done = (m & 512) != 0; g_season_done = (m & 1024) != 0;
   for(int s = 0; s < NSYM; s++) { g_idx_done[s] = (m & (1 << (11 + s))) != 0; g_fomc_done[s] = (m & (1 << (14 + s))) != 0; }
  }

// line 1: day,day_ref,day_trades,stopped,capped,flags,drift_low ; then one line per position:
// ticket,s,entry,hold_min,trail,risk_usd,leg
void SaveState()
  {
   if(!g_live) return;
   int h = FileOpen(StateFile(), FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return;
   FileWriteString(h, StringFormat("%I64d,%.2f,%d,%d,%d,%d,%d\n", (long)g_day, g_day_ref, g_day_trades, (int)g_stopped,
                                   (int)g_capped, Flags(), (int)g_drift_low));
   for(int i = 0; i < ArraySize(g_pos); i++)
      FileWriteString(h, StringFormat("%I64u,%d,%I64d,%d,%.5f,%.2f,%s\n", g_pos[i].ticket, g_pos[i].s, (long)g_pos[i].entry,
                                      g_pos[i].hold_min, g_pos[i].trail, g_pos[i].risk_usd, g_pos[i].leg));
   FileClose(h);
  }

void LoadState()
  {
   if(!FileIsExist(StateFile(), FILE_COMMON)) { Print("[State] no state file - fresh start"); return; }
   int h = FileOpen(StateFile(), FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return;
   string parts[];
   if(!FileIsEnding(h) && StringSplit(FileReadString(h), ',', parts) == 7)
     {
      g_stopped = (int)StringToInteger(parts[3]) != 0;                   // a total stop is permanent
      if((datetime)StringToInteger(parts[0]) == DayOf(TimeCurrent()))   // same server day: keep the day's counters
        {
         g_day = (datetime)StringToInteger(parts[0]); g_day_ref = StringToDouble(parts[1]);
         g_day_trades = (int)StringToInteger(parts[2]); g_capped = (int)StringToInteger(parts[4]) != 0;
         SetFlags((int)StringToInteger(parts[5])); g_drift_low = (int)StringToInteger(parts[6]) != 0;
        }
     }
   int kept = 0;
   while(!FileIsEnding(h))
     {
      if(StringSplit(FileReadString(h), ',', parts) != 7) continue;
      ulong t = (ulong)StringToInteger(parts[0]);
      if(!PositionSelectByTicket(t)) continue;                           // closed while the EA was off
      SPos q;
      q.ticket = t; q.s = (int)StringToInteger(parts[1]); q.entry = (datetime)StringToInteger(parts[2]);
      q.hold_min = (int)StringToInteger(parts[3]); q.trail = StringToDouble(parts[4]); q.risk_usd = StringToDouble(parts[5]);
      q.leg = parts[6];
      int k = ArraySize(g_pos); ArrayResize(g_pos, k + 1); g_pos[k] = q; kept++;
     }
   FileClose(h);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      bool known = false;
      for(int j = 0; j < ArraySize(g_pos); j++) if(g_pos[j].ticket == t) known = true;
      if(!known) PrintFormat("[State] WARNING position %I64u (%s) not in the state file - only its SL protects it", t,
                             PositionGetString(POSITION_SYMBOL));
     }
   PrintFormat("[State] restored: %d open positions, day trades %d, stopped=%d capped=%d", kept, g_day_trades,
               (int)g_stopped, (int)g_capped);
  }

//--------------------------------------------------------------- helpers
int      MinuteOfDay(datetime t) { return (int)((t % 86400) / 60); }
int      Dow(datetime t)         { return (int)((t / 86400 + 4) % 7); }     // 0 = Sunday
datetime DayOf(datetime t)       { return (datetime)(t / 86400 * 86400); }

// Wilder ATR (pandas ewm(alpha=1/n, adjust=False) of the true range) on closed bars ending at `shift`
double WilderATR(string sym, ENUM_TIMEFRAMES tf, int period, int shift, int lookback = 400)
  {
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int n = CopyRates(sym, tf, shift, lookback, r);
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
int DailyState(string sym)
  {
   MqlRates d[];
   ArraySetAsSeries(d, false);
   int n = CopyRates(sym, PERIOD_D1, 1, 500, d);
   if(n < 100) return 0;
   double e20 = EMA(d, n - 1, 20), e50 = EMA(d, n - 1, 50), c = d[n - 1].close;
   if(c > e50 && e20 > e50) return 1;
   if(c < e50 && e20 < e50) return -1;
   return 0;
  }

// drift filter: yesterday's ATR_D in bps <= median of all earlier daily values since InpDriftMedStart (>= 120)
bool DriftLowVol()
  {
   MqlRates d[];
   ArraySetAsSeries(d, false);
   int n = CopyRates(_Symbol, PERIOD_D1, InpDriftMedStart, DayOf(TimeCurrent()) - 1, d);
   if(n < 122) return false;
   double x[];
   ArrayResize(x, n);
   double a = 0;
   for(int i = 0; i < n; i++)
     {
      double tr = d[i].high - d[i].low;
      if(i > 0) tr = MathMax(tr, MathMax(MathAbs(d[i].high - d[i - 1].close), MathAbs(d[i].low - d[i - 1].close)));
      a = (i == 0) ? tr : a + (tr - a) / 14.0;
      x[i] = a / d[i].close * 1e4;
     }
   double last = x[n - 1];
   double h[];
   ArrayResize(h, n - 1);
   for(int i = 0; i < n - 1; i++) h[i] = x[i];
   ArraySort(h);
   int m = n - 1;
   double med = (m % 2 == 1) ? h[m / 2] : 0.5 * (h[m / 2 - 1] + h[m / 2]);
   return last <= med;
  }

int CountOpen(int s, int &dir, double &open_risk, int &nsym)
  {
   dir = 0; open_risk = 0; nsym = 0;
   int k = 0;
   for(int i = ArraySize(g_pos) - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(g_pos[i].ticket)) { ArrayRemove(g_pos, i, 1); continue; }
      open_risk += g_pos[i].risk_usd;
      if(g_pos[i].s == s)
        {
         nsym++;
         dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
        }
      k++;
     }
   return k;
  }

void CloseTicket(int i, string why)
  {
   if(g_trade.PositionClose(g_pos[i].ticket))
      PrintFormat("[Exit] %s %s %s", g_sym[g_pos[i].s], g_pos[i].leg, why);
  }

void CloseAll(string why)
  {
   for(int i = ArraySize(g_pos) - 1; i >= 0; i--) CloseTicket(i, why);
  }

//--------------------------------------------------------------- entry
// claim the (symbol, minute) slot: the first signal of a symbol in a minute wins, even if it is then rejected
bool Claim(int s)
  {
   datetime m = TimeCurrent() / 60 * 60;
   if(g_slot[s] == m) return false;
   g_slot[s] = m;
   return true;
  }

bool Enter(int s, int dir, double sl_dist, int hold_min, double trail, double risk_mult, string leg)
  {
   if(!g_on[s] || !Claim(s)) return false;
   string sym = g_sym[s];
   datetime now = TimeCurrent();
   int tm = MinuteOfDay(now);
   if(g_stopped || g_capped || tm < InpFirstEntryMin || tm > InpLastEntryMin) return false;
   if(g_news.BlockEntry(now) || g_day_trades >= InpMaxTradesDay) return false;
   if(g_live)
     {
      datetime se = SessionEnd(sym, now);
      if(se > 0 && now >= se - 15 * 60) return false;
     }
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double day_pnl = bal - g_day_ref;
   if(day_pnl <= -InpDailySoftPct / 100.0 * g_day_ref) return false;
   int cur_dir, nsym; double open_risk;
   int nopen = CountOpen(s, cur_dir, open_risk, nsym);
   if(nopen >= InpMaxPositions || (nsym > 0 && cur_dir != dir)) return false;       // no hedging per symbol
   if(sl_dist <= 0) return false;
   double dd = (InpInitialBalance - bal) / InpInitialBalance * 100.0;
   double rp;
   if(InpCushionStartPct > 0)
     {
      double k = MathMin(1.0, MathMax(0.1, (InpTotalStopPct - dd) / (InpTotalStopPct - InpCushionStartPct)));
      rp = InpRiskPct * k;
      if(k < 1.0) PrintFormat("[Cushion] static DD %.2f %% -> risk x %.2f", dd, k);
     }
   else
      rp = (dd < InpTotalDeriskPct) ? InpRiskPct : InpRiskPct * 0.5;
   double base = InpRiskOnInitial ? InpInitialBalance : bal;
   double room = InpDailyHardPct / 100.0 * g_day_ref + day_pnl - open_risk;
   double risk = MathMin(MathMin(rp * MathMin(risk_mult, 1.0) / 100.0 * base, room), InpMaxOpenRiskPct / 100.0 * base - open_risk);
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP), vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double loss1 = LossPerLot(sym, dir, sl_dist);
   if(loss1 <= 0) return false;
   double per_lot = loss1 + g_comm[s];
   double lots = MathFloor(risk / per_lot / step + 1e-9) * step;
   if(lots < vmin - 1e-9) return false;
   lots = NormalizeDouble(lots, 2);
   int dg = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   bool ok;
   string cmt = leg;
   g_trade.SetTypeFillingBySymbol(sym);
   if(dir > 0)
     {
      double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
      ok = g_trade.Buy(lots, sym, 0, NormalizeDouble(ask - sl_dist, dg), 0, cmt);
     }
   else
     {
      double bid = SymbolInfoDouble(sym, SYMBOL_BID);
      ok = g_trade.Sell(lots, sym, 0, NormalizeDouble(bid + sl_dist, dg), 0, cmt);
     }
   if(!ok || (g_trade.ResultRetcode() != TRADE_RETCODE_DONE && g_trade.ResultRetcode() != TRADE_RETCODE_PLACED))
     {
      PrintFormat("[Entry-fail] %s %s ret=%d", sym, leg, g_trade.ResultRetcode());
      if(g_live && g_trade.ResultRetcode() == TRADE_RETCODE_CLIENT_DISABLES_AT)
         Alert("XauIdxPortfolio: order rejected - Algo Trading is disabled (EA properties > Common > Allow Algo Trading, and the toolbar button)");
      return false;
     }
   SPos p;
   p.ticket = g_trade.ResultOrder(); p.s = s; p.entry = now; p.hold_min = hold_min; p.trail = trail;
   p.risk_usd = lots * per_lot; p.leg = leg;
   if(!PositionSelectByTicket(p.ticket))
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong t = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) == InpMagic && PositionGetString(POSITION_SYMBOL) == sym &&
            PositionGetString(POSITION_COMMENT) == cmt) { p.ticket = t; break; }
        }
   int k = ArraySize(g_pos);
   ArrayResize(g_pos, k + 1);
   g_pos[k] = p;
   g_day_trades++;
   if(InpExportTrades)
     {
      double px = dir > 0 ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID);
      int n = ArraySize(g_entry_log);
      ArrayResize(g_entry_log, n + 1, 4096);
      g_entry_log[n] = StringFormat("%I64u,%s,%s,%d,%s,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.4f,%.3f",
                                    p.ticket, sym, leg, dir, TimeToString(now, TIME_DATE | TIME_SECONDS), px,
                                    dir > 0 ? px - sl_dist : px + sl_dist, sl_dist, lots, p.risk_usd, bal,
                                    p.risk_usd / bal * 100.0, rp * MathMin(risk_mult, 1.0) / InpRiskPct);
      if(g_live) AppendEntry(g_entry_log[n]);
     }
   SaveState();
   PrintFormat("[Entry] %s %s %s %.2f lots sl=%.2f hold=%d trail=%.2f", sym, leg, dir > 0 ? "BUY" : "SELL", lots, sl_dist, hold_min, trail);
   return true;
  }

//--------------------------------------------------------------- XAUUSD legs (book order matters for slots)
void XauOnH4()
  {
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
      Enter(0, 1, 2.0 * a, 240 * 240, 6.0 * a, 1.0, "trendH4");
     }
  }

void XauOnM15(datetime now)
  {
   MqlRates b[];
   ArraySetAsSeries(b, false);
   datetime day = DayOf(now);
   int n = CopyRates(_Symbol, PERIOD_M15, day, now - 1, b);          // today's CLOSED M15 bars
   if(n < 1) return;
   MqlRates last = b[n - 1];
   if(last.time + 15 * 60 > now) { n--; if(n < 1) return; last = b[n - 1]; }
   int tm_open = MinuteOfDay(last.time);
   double o = b[0].open, hi = b[0].high, lo = b[0].low;
   for(int i = 0; i < n; i++) { hi = MathMax(hi, b[i].high); lo = MathMin(lo, b[i].low); }
   double atr_d = WilderATR(_Symbol, PERIOD_D1, 14, 1);
   int st = DailyState(_Symbol);
   MqlRates d[];
   ArraySetAsSeries(d, true);
   if(CopyRates(_Symbol, PERIOD_D1, 1, 8, d) < 8) return;              // d[0] = yesterday
   double pdh = d[0].high, pdl = d[0].low, prev_rng = d[0].high - d[0].low;
   double pos = (hi > lo) ? (last.close - lo) / (hi - lo) : 0.5;
   // 2. tday (18:00 bar)
   if(!g_tday_done && tm_open == 1080 && st != 0 && atr_d > 0)
     {
      g_tday_done = true;
      if((last.close - o) * st >= 0.3 * atr_d && ((st == 1 && pos >= 0.75) || (st == -1 && pos <= 0.25)))
         Enter(0, st, 1.0 * atr_d, 330, 1.5 * atr_d, 1.0, "tday");
     }
   // 6. lw: 10:00..20:00, first close beyond open +/- 0.4 prev range
   if(!g_lw_done && tm_open >= 600 && tm_open <= 1200 && prev_rng > 0 && atr_d > 0)
     {
      int dir = 0;
      if(last.close > o + 0.4 * prev_rng) dir = 1; else if(last.close < o - 0.4 * prev_rng) dir = -1;
      if(dir != 0) { g_lw_done = true; Enter(0, dir, 1.0 * atr_d, 1440, 1.5 * atr_d, 1.0, "lw"); }
     }
   // 7./8. inside / nr7 breakouts with the D1 state, 01:05..20:00
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
         if(inside && !g_inside_done) { g_inside_done = true; if(dir == st) Enter(0, dir, risk, 1440, 1.5 * risk, 1.0, "inside"); }
         if(nr7 && !g_nr7_done) { g_nr7_done = true; if(dir == st) Enter(0, dir, risk, 1440, 1.5 * risk, 1.0, "nr7"); }
        }
     }
   // 10. tday900 (15:00 bar)
   if(!g_tday900_done && tm_open == 900 && st != 0 && atr_d > 0)
     {
      g_tday900_done = true;
      if((last.close - o) * st >= 0.3 * atr_d && ((st == 1 && pos >= 0.75) || (st == -1 && pos <= 0.25)))
         Enter(0, st, 1.0 * atr_d, 510, 1.5 * atr_d, 1.0, "tday900");
     }
  }

void XauOnM1(datetime now)
  {
   int tm = MinuteOfDay(now), dw = Dow(now);
   double atr_d = WilderATR(_Symbol, PERIOD_D1, 14, 1);
   MqlRates y[];
   ArraySetAsSeries(y, true);
   bool have_y = CopyRates(_Symbol, PERIOD_D1, 1, 1, y) == 1;
   // 4. friday: Friday 01:05, SL 1.5 ATR_D, hold 1315
   if(!g_friday_done && tm >= 65 && dw == 5)
     { g_friday_done = true; Enter(0, 1, 1.5 * atr_d, 1315, 2.25 * atr_d, 1.0, "friday"); }
   // 11. strong_close: yesterday clv > 0.6 -> 01:05 long, SL 1 ATR_D, hold 1360
   if(!g_sc_done && tm >= 65 && have_y)
     {
      g_sc_done = true;
      double rng = y[0].high - y[0].low;
      double clv = rng > 0 ? ((y[0].close - y[0].low) - (y[0].high - y[0].close)) / rng : 0;
      if(clv > 0.6) Enter(0, 1, 1.0 * atr_d, 1360, 1.5 * atr_d, 1.0, "strong_close");
     }
   // 12. season: Jan / Jul / Aug every day 01:07, half risk, SL 1 ATR_D, hold 1360
   if(!g_season_done && tm >= 67)
     {
      g_season_done = true;
      MqlDateTime a;
      TimeToStruct(now, a);
      if(a.mon == 1 || a.mon == 7 || a.mon == 8) Enter(0, 1, 1.0 * atr_d, 1360, 1.5 * atr_d, 0.5, "season");
     }
   // 5. tom: first trading day of the month 01:10, SL 2 ATR_D, hold 3 days - 60 min
   if(!g_tom_done && tm >= 70 && have_y)
     {
      g_tom_done = true;
      MqlDateTime a, b;
      TimeToStruct(now, a);
      TimeToStruct(y[0].time, b);
      if(a.mon != b.mon) Enter(0, 1, 2.0 * atr_d, 3 * 1440 - 60, 0, 1.0, "tom");
     }
   // 3. drift: Tue-Fri 01:15, low-vol days only, SL 4 ATR_H1, hold 475
   if(!g_drift_done && tm >= 75 && dw >= 2 && dw <= 5)
     {
      g_drift_done = true;
      if(g_drift_low)
        {
         double a1 = WilderATR(_Symbol, PERIOD_H1, 14, 1);
         Enter(0, 1, 4.0 * a1, 475, 6.0 * a1, 1.0, "drift");
        }
     }
   // 9. fri_close: Friday 23:05, SL 4 ATR_M15, hold 50
   if(!g_fric_done && tm >= 1385 && dw == 5)
     { g_fric_done = true; Enter(0, 1, 4.0 * WilderATR(_Symbol, PERIOD_M15, 14, 1), 50, 0, 1.0, "fri_close"); }
  }

//--------------------------------------------------------------- index legs (NAS100, DJ30): same rules on both
bool IsLastWeekdayOfMonth(datetime now)
  {
   datetime nxt = DayOf(now) + 86400;
   while(Dow(nxt) == 0 || Dow(nxt) == 6) nxt += 86400;
   MqlDateTime a, b;
   TimeToStruct(now, a); TimeToStruct(nxt, b);
   return a.mon != b.mon;
  }

void IdxOnM1(int s, datetime now)
  {
   if(!g_on[s]) return;
   string sym = g_sym[s];
   int tm = MinuteOfDay(now), dw = Dow(now);
   // daily legs at 01:05 in the order mon, dip_low20, dip_clv, hi20, (prefomc), tom
   if(!g_idx_done[s] && tm >= 65)
     {
      MqlRates d[];
      ArraySetAsSeries(d, true);
      if(CopyRates(sym, PERIOD_D1, 1, 20, d) < 20) return;           // d[0] = yesterday
      g_idx_done[s] = true;
      double atr_d = WilderATR(sym, PERIOD_D1, 14, 1);
      double hi20 = d[0].high, lo20 = d[0].low;
      for(int i = 0; i < 20; i++) { hi20 = MathMax(hi20, d[i].high); lo20 = MathMin(lo20, d[i].low); }
      double pos20 = hi20 > lo20 ? (d[0].close - lo20) / (hi20 - lo20) : 0.5;
      double rng = d[0].high - d[0].low;
      double clv = rng > 0 ? ((d[0].close - d[0].low) - (d[0].high - d[0].close)) / rng : 0;
      int hold = 1410 - 65;
      if(dw == 1) Enter(s, 1, 1.5 * atr_d, hold, 2.25 * atr_d, 1.0, "mon");
      if(pos20 < 0.1) Enter(s, 1, 1.0 * atr_d, hold, 1.5 * atr_d, 1.0, "dip_low20");
      if(clv < -0.6) Enter(s, 1, 1.0 * atr_d, hold, 1.5 * atr_d, 1.0, "dip_clv");
      if(pos20 > 0.9) Enter(s, 1, 1.0 * atr_d, hold, 1.5 * atr_d, 1.0, "hi20");
      if(IsLastWeekdayOfMonth(now)) Enter(s, 1, 2.0 * atr_d, 4 * 1440, 0, 1.0, "tom");
     }
   // prefomc: 24 h before the FOMC decision, SL 1.5 ATR_D, hold 1500 (flattened by the news rule)
   if(!g_fomc_done[s])
      for(int i = 0; i < ArraySize(g_fomc); i++)
        {
         datetime ent = g_fomc[i] - 86400;
         if(now >= ent && now < ent + 60)
           {
            g_fomc_done[s] = true;
            double atr_d = WilderATR(sym, PERIOD_D1, 14, 1);
            Enter(s, 1, 1.5 * atr_d, 1500, 2.25 * atr_d, 1.0, "prefomc");
            break;
           }
        }
  }

//--------------------------------------------------------------- management
void Manage(datetime now, bool new_m1)
  {
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(ArraySize(g_pos) > 0 && eq - g_day_ref <= -InpDailyHardPct / 100.0 * g_day_ref) { CloseAll("daily hard guard"); return; }
   if(InpDayProfitCap > 0 && ArraySize(g_pos) > 0 && eq - g_day_ref >= InpDayProfitCap / 100.0 * g_day_ref)
     { CloseAll("day profit cap"); g_capped = true; SaveState(); return; }
   if(g_news.MustFlatten(now)) { CloseAll("news flatten"); return; }
   if(Dow(now) == 5 && MinuteOfDay(now) >= 1435) { CloseAll("weekend"); return; }
   if(g_live)
      for(int i = ArraySize(g_pos) - 1; i >= 0; i--)
        {
         datetime se = SessionEnd(g_sym[g_pos[i].s], now);
         if(se <= 0 || now < se - 5 * 60) continue;
         // Friday: nothing is carried over the weekend; other days: close positions whose planned exit falls in the
         // session break (research exits at 23:30 on a broker trading until 23:57)
         if(Dow(now) == 5 || g_pos[i].entry + g_pos[i].hold_min * 60 < se + 4 * 3600)
            CloseTicket(i, Dow(now) == 5 ? "weekend (session end)" : "session end");
        }
   for(int i = ArraySize(g_pos) - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(g_pos[i].ticket)) { ArrayRemove(g_pos, i, 1); continue; }
      if(now >= g_pos[i].entry + g_pos[i].hold_min * 60) { CloseTicket(i, "max hold"); continue; }
      if(new_m1 && g_pos[i].trail > 0)
        {
         string sym = g_sym[g_pos[i].s];
         int dg = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
         double pt = SymbolInfoDouble(sym, SYMBOL_POINT);
         double sl = PositionGetDouble(POSITION_SL), tp = PositionGetDouble(POSITION_TP);
         if(iTime(sym, PERIOD_M1, 1) < g_pos[i].entry) continue;        // only bars after the entry
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
           {
            double nsl = NormalizeDouble(iHigh(sym, PERIOD_M1, 1) - g_pos[i].trail, dg);
            if(nsl > sl + pt && nsl < SymbolInfoDouble(sym, SYMBOL_BID)) g_trade.PositionModify(g_pos[i].ticket, nsl, tp);
           }
         else
           {
            double spr = SymbolInfoDouble(sym, SYMBOL_ASK) - SymbolInfoDouble(sym, SYMBOL_BID);
            double nsl = NormalizeDouble(iLow(sym, PERIOD_M1, 1) + spr + g_pos[i].trail, dg);
            if(nsl < sl - pt && nsl > SymbolInfoDouble(sym, SYMBOL_ASK)) g_trade.PositionModify(g_pos[i].ticket, nsl, tp);
           }
        }
     }
  }

bool LoadFomc()
  {
   int h = FileOpen(InpFomcFile, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) { PrintFormat("[FOMC] cannot open %s", InpFomcFile); return false; }
   ArrayResize(g_fomc, 0, 256);
   while(!FileIsEnding(h))
     {
      string s = FileReadString(h);
      if(StringLen(s) < 10) continue;
      datetime t = StringToTime(s);
      if(t <= 0) continue;
      int k = ArraySize(g_fomc);
      ArrayResize(g_fomc, k + 1, 256);
      g_fomc[k] = t;
     }
   FileClose(h);
   PrintFormat("[FOMC] %d decisions loaded", ArraySize(g_fomc));
   return ArraySize(g_fomc) > 0;
  }

//--------------------------------------------------------------- events
int OnInit()
  {
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(100);
   g_sym[0] = _Symbol; g_sym[1] = InpNasSymbol; g_sym[2] = InpDjSymbol;
   g_on[0] = InpTradeXau; g_on[1] = InpTradeNas; g_on[2] = InpTradeDj;
   g_comm[0] = InpCommXau; g_comm[1] = InpCommIdx; g_comm[2] = InpCommIdx;
   for(int s = 1; s < NSYM; s++)
      if(g_on[s] && !SymbolSelect(g_sym[s], true)) { PrintFormat("[Init] symbol %s not available", g_sym[s]); return INIT_FAILED; }
   if(!g_news.Load(InpNewsFile, InpNewsBefore, InpNewsAfter, InpNewsFlatten)) { Print("[Init] news file missing - refusing to trade"); return INIT_FAILED; }
   if((g_on[1] || g_on[2]) && !LoadFomc()) { Print("[Init] FOMC file missing - refusing to trade"); return INIT_FAILED; }
   g_live = !(bool)MQLInfoInteger(MQL_TESTER) && !(bool)MQLInfoInteger(MQL_OPTIMIZATION);
   if(g_live)
     {
      if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
         Alert("XauIdxPortfolio: 'Allow Algo Trading' is OFF in this EA's properties (Common tab) - orders will be rejected (10027)");
      if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
         Alert("XauIdxPortfolio: the terminal's Algo Trading button is OFF - orders will be rejected (10027)");
      if(InpDemoOnly && AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_DEMO)
        { Print("[Init] InpDemoOnly: this is not a demo account - refusing to run"); return INIT_FAILED; }
      for(int s = 0; s < NSYM; s++)
        {
         if(!g_on[s]) continue;
         if(SymbolInfoInteger(g_sym[s], SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
           { PrintFormat("[Init] trading is disabled for %s on this server - switch it off in the inputs", g_sym[s]); return INIT_FAILED; }
         datetime se = SessionEnd(g_sym[s], TimeCurrent());
         PrintFormat("[Init] %s session end today %s", g_sym[s], se > 0 ? TimeToString(se, TIME_MINUTES) : "24:00 / unknown");
        }
      datetime gmt = TimeGMT();
      int off = (int)MathRound((double)(TimeTradeServer() - gmt) / 1800.0) * 1800, expo = ExpectedServerOffset(gmt);
      PrintFormat("[Init] server offset GMT%+.1f h, expected GMT%+.1f h (NY + 7 h)", off / 3600.0, expo / 3600.0);
      if(InpCheckServerTime && off != expo)
        { Print("[Init] server time zone differs from the research convention (sessions, news, daily bars) - refusing to trade"); return INIT_FAILED; }
      if(g_news.LastEvent() < TimeCurrent()) { Print("[Init] news file does not cover the future - refusing to trade"); return INIT_FAILED; }
      if(g_news.LastEvent() < TimeCurrent() + 21 * 86400)
         PrintFormat("[Init] WARNING news file ends %s - update it (tools/export_news.py)", TimeToString(g_news.LastEvent()));
      if((g_on[1] || g_on[2]) && g_fomc[ArraySize(g_fomc) - 1] < TimeCurrent())
         Print("[Init] WARNING no future FOMC decision in the FOMC file - prefomc leg inactive until it is updated");
      for(int s = 0; s < NSYM; s++)
         if(g_on[s]) PrintFormat("[Init] %s loss of 1 lot per 1.0 price move: OrderCalcProfit %.2f vs tick value %.2f", g_sym[s],
                                 LossPerLot(g_sym[s], 1, 1.0), SymbolInfoDouble(g_sym[s], SYMBOL_TRADE_TICK_VALUE_LOSS) / SymbolInfoDouble(g_sym[s], SYMBOL_TRADE_TICK_SIZE));
      for(int s = 0; s < NSYM; s++)
         if(g_on[s]) PrintFormat("[Init] %s tick value %.5f, tick size %.5f, vol min %.2f step %.2f, filling %d", g_sym[s],
                                 SymbolInfoDouble(g_sym[s], SYMBOL_TRADE_TICK_VALUE_LOSS), SymbolInfoDouble(g_sym[s], SYMBOL_TRADE_TICK_SIZE),
                                 SymbolInfoDouble(g_sym[s], SYMBOL_VOLUME_MIN), SymbolInfoDouble(g_sym[s], SYMBOL_VOLUME_STEP),
                                 (int)SymbolInfoInteger(g_sym[s], SYMBOL_FILLING_MODE));
      LoadState();
     }
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   SaveState();
   if(!InpExportTrades)
      return;
   int h = g_live ? INVALID_HANDLE : FileOpen(InpExportPrefix + "_entries.csv", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h != INVALID_HANDLE)                            // tester: whole entry log once (live appends at entry time)
     {
      FileWriteString(h, "position,symbol,leg,dir,time,price_at_signal,sl_price,sl_dist,lots,risk_usd,balance,risk_pct,risk_mult_used\n");
      for(int i = 0; i < ArraySize(g_entry_log); i++) FileWriteString(h, g_entry_log[i] + "\n");
      FileClose(h);
     }
   ExportDeals();
  }

void ExportDeals()
  {
   if(!HistorySelect(0, TimeCurrent() + 86400)) return;
   int h = FileOpen(InpExportPrefix + "_deals.csv", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) return;
   FileWriteString(h, "deal,position,time,symbol,type,entry,volume,price,profit,commission,swap,reason,sl,comment\n");
   for(int i = 0; i < HistoryDealsTotal(); i++)
     {
      ulong d = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(d, DEAL_MAGIC) != InpMagic) continue;
      FileWriteString(h, StringFormat("%I64u,%I64d,%s,%s,%d,%d,%.2f,%.5f,%.2f,%.2f,%.2f,%d,%.5f,%s\n", d,
                                      HistoryDealGetInteger(d, DEAL_POSITION_ID),
                                      TimeToString((datetime)HistoryDealGetInteger(d, DEAL_TIME), TIME_DATE | TIME_SECONDS),
                                      HistoryDealGetString(d, DEAL_SYMBOL), (int)HistoryDealGetInteger(d, DEAL_TYPE),
                                      (int)HistoryDealGetInteger(d, DEAL_ENTRY), HistoryDealGetDouble(d, DEAL_VOLUME),
                                      HistoryDealGetDouble(d, DEAL_PRICE), HistoryDealGetDouble(d, DEAL_PROFIT),
                                      HistoryDealGetDouble(d, DEAL_COMMISSION), HistoryDealGetDouble(d, DEAL_SWAP),
                                      (int)HistoryDealGetInteger(d, DEAL_REASON), HistoryDealGetDouble(d, DEAL_SL),
                                      HistoryDealGetString(d, DEAL_COMMENT)));
     }
   FileClose(h);
   PrintFormat("[Export] %d entries, deals written to Common/Files/%s_*.csv", ArraySize(g_entry_log), InpExportPrefix);
  }

void OnTick()
  {
   if(g_live && InpDemoOnly && AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_DEMO)
     { Print("[Guard] account is no longer a demo - EA removed"); ExpertRemove(); return; }
   datetime now = TimeCurrent();
   datetime day = DayOf(now);
   if(day != g_day)
     {
      g_day = day;
      g_day_ref = MathMax(AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));
      g_day_trades = 0; g_capped = false;
      g_lw_done = g_inside_done = g_nr7_done = g_tday_done = g_tday900_done = g_drift_done = g_friday_done = false;
      g_tom_done = g_fric_done = g_sc_done = g_season_done = false;
      for(int s = 0; s < NSYM; s++) { g_idx_done[s] = false; g_fomc_done[s] = false; }
      g_drift_low = (Dow(now) >= 2 && Dow(now) <= 5) ? DriftLowVol() : false;
      if(g_live) { if(InpExportTrades) ExportDeals(); SaveState(); }
     }
   if(!g_stopped && (InpInitialBalance - AccountInfoDouble(ACCOUNT_EQUITY)) / InpInitialBalance * 100.0 >= InpTotalStopPct)
     {
      g_stopped = true; CloseAll("total stop"); SaveState();
      Print("[Guard] TOTAL STOP - trading disabled");
     }
   datetime m1 = iTime(_Symbol, PERIOD_M1, 0);
   bool new_m1 = (m1 != g_last_m1);
   Manage(now, new_m1);
   if(!new_m1 || g_stopped) return;
   g_last_m1 = m1;
   // book order: trendH4, tday, drift, friday, tom, lw, inside, nr7, fri_close, tday900, strong_close, season
   // (collisions only matter within one minute; H4 and M15 legs are evaluated before the M1 legs)
   datetime h4 = iTime(_Symbol, PERIOD_H4, 0);
   if(h4 != g_last_h4) { g_last_h4 = h4; if(g_on[0]) XauOnH4(); }
   datetime m15 = iTime(_Symbol, PERIOD_M15, 0);
   if(m15 != g_last_m15) { g_last_m15 = m15; if(g_on[0]) XauOnM15(now); }
   if(g_on[0]) XauOnM1(now);
   IdxOnM1(1, now);
   IdxOnM1(2, now);
   if(g_live) SaveState();
  }

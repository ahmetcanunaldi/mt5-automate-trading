//+------------------------------------------------------------------+
//| PropGuard.mqh - FundingPips-style risk guards (internal limits)  |
//| Mirrors research/engine.py:                                      |
//|  - day reference = max(balance, equity) at the first tick of the |
//|    server day                                                    |
//|  - soft daily stop (no new entries), hard daily stop (close all) |
//|  - total drawdown: de-risk, then stop trading for good           |
//|  - lot size from SL distance, rounded DOWN, never above risk %   |
//+------------------------------------------------------------------+
#property strict

class CPropGuard
  {
private:
   double            m_initial;
   double            m_risk_pct;
   double            m_soft_pct;
   double            m_hard_pct;
   double            m_derisk_pct;
   double            m_stop_pct;
   double            m_comm_per_lot;
   int               m_max_trades_day;
   datetime          m_day;
   double            m_day_ref;
   int               m_day_trades;
   bool              m_stopped;

public:
   void              Init(double initial, double risk_pct, double soft, double hard, double derisk, double stop,
                          double comm, int max_trades)
     {
      m_initial = initial; m_risk_pct = risk_pct; m_soft_pct = soft; m_hard_pct = hard;
      m_derisk_pct = derisk; m_stop_pct = stop; m_comm_per_lot = comm; m_max_trades_day = max_trades;
      m_day = 0; m_day_ref = 0; m_day_trades = 0; m_stopped = false;
     }

   //--- call on every tick
   void              OnTickUpdate()
     {
      datetime d = (datetime)(TimeCurrent() / 86400 * 86400);
      if(d != m_day)
        {
         m_day = d;
         m_day_ref = MathMax(AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));
         m_day_trades = 0;
         PrintFormat("[PropGuard] new server day %s ref=%.2f", TimeToString(d, TIME_DATE), m_day_ref);
        }
      if(!m_stopped && TotalDDPct() >= m_stop_pct)
        {
         m_stopped = true;
         PrintFormat("[PropGuard] TOTAL STOP reached (dd=%.2f%%) - trading disabled", TotalDDPct());
        }
     }

   double            DayRef()        { return m_day_ref; }
   double            DayPnL()        { return AccountInfoDouble(ACCOUNT_BALANCE) - m_day_ref; }
   double            DayPnLEquity()  { return AccountInfoDouble(ACCOUNT_EQUITY) - m_day_ref; }
   double            TotalDDPct()    { return (m_initial - AccountInfoDouble(ACCOUNT_EQUITY)) / m_initial * 100.0; }
   bool              Stopped()       { return m_stopped; }
   int               DayTrades()     { return m_day_trades; }
   void              CountTrade()    { m_day_trades++; }

   //--- hard daily guard: true -> close everything now
   bool              HardBreach()
     {
      return DayPnLEquity() <= -m_hard_pct / 100.0 * m_day_ref;
     }

   //--- may a new position be opened?
   bool              EntryAllowed(string &why)
     {
      if(m_stopped)                     { why = "total stop"; return false; }
      if(m_day_trades >= m_max_trades_day) { why = "max trades/day"; return false; }
      if(DayPnL() <= -m_soft_pct / 100.0 * m_day_ref) { why = "daily soft stop"; return false; }
      return true;
     }

   //--- lots for a stop distance in price units; 0 when not tradable
   double            Lots(string sym, double sl_dist, bool min_lot_only)
     {
      double tv = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_LOSS);
      double ts = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
      double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
      double vstep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
      if(tv <= 0 || ts <= 0 || sl_dist <= 0)
         return 0.0;
      double money_per_lot = sl_dist / ts * tv + m_comm_per_lot;
      double bal = AccountInfoDouble(ACCOUNT_BALANCE);
      double rp = (((m_initial - bal) / m_initial * 100.0) < m_derisk_pct) ? m_risk_pct : m_risk_pct * 0.5;
      double risk = rp / 100.0 * bal;
      double room = m_hard_pct / 100.0 * m_day_ref + DayPnL();
      risk = MathMin(risk, room);
      double lots = MathFloor(risk / money_per_lot / vstep + 1e-9) * vstep;
      if(min_lot_only && lots >= vmin)
         lots = vmin;
      if(lots < vmin)
         return 0.0;
      return NormalizeDouble(lots, 2);
     }
  };

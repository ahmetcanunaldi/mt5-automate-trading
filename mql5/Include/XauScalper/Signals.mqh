//+------------------------------------------------------------------+
//| Signals.mqh - M15 decision logic, mirrors research/features.py + |
//| research/strategies/rules.py (legs of research/portfolio.py).    |
//| Evaluated once per closed M15 bar (bar index 1).                 |
//+------------------------------------------------------------------+
#property strict

#define MIN_TP_USD 6.0

struct SSignal
  {
   int               dir;       // +1 long, -1 short, 0 none
   double            sl;        // price distance
   double            tp;        // price distance
   int               hold_min;
   double            trail;     // price distance, 0 = off
   string            leg;
  };

struct SLegOrb
  {
   bool              on;
   int               or_start;  // minute of server day
   int               or_bars;
   int               end_min;
   double            sl_atr, rr, trail_atr, min_rng_atr, max_rng_atr;
   int               hold_min;
   int               trend;     // 0 none, 1 with, 2 long_only
  };

struct SLegDonchian
  {
   bool              on;
   int               n, start_min, end_min, per_day, hold_min, trend;
   double            sl_atr, rr, trail_atr, min_width_atr;
  };

struct SLegPdb
  {
   bool              on;
   int               start_min, end_min, hold_min, trend;
   double            sl_atr, rr, trail_atr, buffer_atr;
  };

class CSignals
  {
private:
   string            m_sym;
   // per-day "first signal" bookkeeping (mirrors _first_per_day on raw conditions)
   datetime          m_day;
   int               m_orb_l, m_orb_s, m_don_l, m_don_s, m_pdb_l, m_pdb_s;

   double            WilderATR(const MqlRates &r[], int n_period)
     {
      // r[] is chronological (oldest first); returns ATR at the last element
      int n = ArraySize(r);
      double a = 0;
      for(int i = 0; i < n; i++)
        {
         double tr = r[i].high - r[i].low;
         if(i > 0)
            tr = MathMax(tr, MathMax(MathAbs(r[i].high - r[i - 1].close), MathAbs(r[i].low - r[i - 1].close)));
         a = (i == 0) ? tr : a + (tr - a) / n_period;
        }
      return a;
     }

   double            EMA(const MqlRates &r[], int span)
     {
      double alpha = 2.0 / (span + 1.0), e = 0;
      int n = ArraySize(r);
      for(int i = 0; i < n; i++)
         e = (i == 0) ? r[i].close : e + alpha * (r[i].close - e);
      return e;
     }

   int               MinuteOfDay(datetime t) { return (int)((t % 86400) / 60); }

public:
   // context of the last evaluation (for logging)
   double            atr, h1_e50, h1_e200, h1_close;

   void              Init(string sym) { m_sym = sym; m_day = 0; }

   bool              TrendOk(int mode, int dir)
     {
      if(mode == 0) return true;
      if(mode == 2) return dir == 1;
      bool up = (h1_e50 > h1_e200) && (h1_close > h1_e50);
      bool dn = (h1_e50 < h1_e200) && (h1_close < h1_e50);
      return dir == 1 ? up : dn;
     }

   void              Build(SSignal &s, int dir, double sl, double rr, double trail_atr, int hold, string leg)
     {
      sl = MathMax(sl, 2.0);
      if(sl > 40.0) { s.dir = 0; return; }
      s.dir = dir; s.sl = sl; s.tp = MathMax(MIN_TP_USD, rr * sl);
      s.trail = trail_atr > 0 ? trail_atr * atr : 0.0;
      s.hold_min = hold; s.leg = leg;
     }

   //--- evaluate the just-closed M15 bar; legs in priority order ORB > Donchian > PDB
   bool              Evaluate(SLegOrb &orb, SLegDonchian &don, SLegPdb &pdb, SSignal &out)
     {
      out.dir = 0;
      MqlRates m15[];
      ArraySetAsSeries(m15, false);
      int need = 400;
      if(CopyRates(m_sym, PERIOD_M15, 1, need, m15) != need)
         return false;
      int last = need - 1;                      // the closed bar being evaluated
      MqlRates b = m15[last];
      datetime day = (datetime)(b.time / 86400 * 86400);
      if(day != m_day)
        {
         m_day = day; m_orb_l = m_orb_s = m_don_l = m_don_s = m_pdb_l = m_pdb_s = 0;
        }
      int tmin = MinuteOfDay(b.time);
      atr = WilderATR(m15, 14);
      // H1 context: last H1 bar closed at or before this M15 bar's close
      datetime close_t = b.time + 15 * 60;
      MqlRates h1[];
      ArraySetAsSeries(h1, false);
      int nh = CopyRates(m_sym, PERIOD_H1, 0, 1200, h1);
      if(nh < 400)
         return false;
      int k = nh - 1;
      while(k >= 0 && h1[k].time + 3600 > close_t)
         k--;
      if(k < 300)
         return false;
      MqlRates hh[];
      ArrayCopy(hh, h1, 0, 0, k + 1);
      h1_e50 = EMA(hh, 50);
      h1_e200 = EMA(hh, 200);
      h1_close = hh[k].close;

      // previous server day high/low from D1
      MqlRates d1[];
      ArraySetAsSeries(d1, true);
      if(CopyRates(m_sym, PERIOD_D1, b.time, 2, d1) != 2)
         return false;
      // d1[0] = day of bar b, d1[1] = previous day
      double dh = d1[1].high, dl = d1[1].low;
      if(d1[0].time != day)
         return false;

      // today's bars (for ORB and Asia range)
      double orh = -DBL_MAX, orl = DBL_MAX;
      int orb_end = orb.or_start + 15 * orb.or_bars;
      for(int i = last; i >= 0 && m15[i].time >= day; i--)
        {
         int tm = MinuteOfDay(m15[i].time);
         if(tm >= orb.or_start && tm < orb_end)
           {
            orh = MathMax(orh, m15[i].high);
            orl = MathMin(orl, m15[i].low);
           }
        }

      SSignal cand;
      cand.dir = 0;
      //--- leg 1: NY opening-range breakout
      if(orb.on && orh > -DBL_MAX && tmin >= orb_end && tmin < orb.end_min)
        {
         double rng = orh - orl;
         bool ok = rng > orb.min_rng_atr * atr && rng < orb.max_rng_atr * atr;
         bool cl = ok && b.close > orh && TrendOk(orb.trend, 1);
         bool cs = ok && b.close < orl && TrendOk(orb.trend, -1);
         if(cl) m_orb_l++;
         if(cs) m_orb_s++;
         if(cl && m_orb_l == 1)
            Build(cand, 1, orb.sl_atr * atr, orb.rr, orb.trail_atr, orb.hold_min, "ny_orb");
         else if(cs && m_orb_s == 1)
            Build(cand, -1, orb.sl_atr * atr, orb.rr, orb.trail_atr, orb.hold_min, "ny_orb");
        }
      //--- leg 2: Donchian breakout (US session)
      if(don.on && tmin >= don.start_min && tmin < don.end_min)
        {
         double hi = -DBL_MAX, lo = DBL_MAX;
         for(int i = last - don.n; i < last; i++)
           {
            hi = MathMax(hi, m15[i].high);
            lo = MathMin(lo, m15[i].low);
           }
         bool ok = (hi - lo) > don.min_width_atr * atr;
         bool cl = ok && b.close > hi && b.open <= hi && TrendOk(don.trend, 1);
         bool cs = ok && b.close < lo && b.open >= lo && TrendOk(don.trend, -1);
         if(cl) m_don_l++;
         if(cs) m_don_s++;
         if(cand.dir == 0)
           {
            if(cl && m_don_l <= don.per_day)
               Build(cand, 1, don.sl_atr * atr, don.rr, don.trail_atr, don.hold_min, "donchian_us");
            else if(cs && m_don_s <= don.per_day)
               Build(cand, -1, don.sl_atr * atr, don.rr, don.trail_atr, don.hold_min, "donchian_us");
           }
        }
      //--- leg 3: previous-day high/low break
      if(pdb.on && tmin >= pdb.start_min && tmin < pdb.end_min)
        {
         double lh = dh + pdb.buffer_atr * atr, ll = dl - pdb.buffer_atr * atr;
         bool cl = b.close > lh && b.open <= lh && TrendOk(pdb.trend, 1);
         bool cs = b.close < ll && b.open >= ll && TrendOk(pdb.trend, -1);
         if(cl) m_pdb_l++;
         if(cs) m_pdb_s++;
         if(cand.dir == 0)
           {
            if(cl && m_pdb_l == 1)
               Build(cand, 1, pdb.sl_atr * atr, pdb.rr, pdb.trail_atr, pdb.hold_min, "pdb");
            else if(cs && m_pdb_s == 1)
               Build(cand, -1, pdb.sl_atr * atr, pdb.rr, pdb.trail_atr, pdb.hold_min, "pdb");
           }
        }
      out = cand;
      return out.dir != 0;
     }
  };

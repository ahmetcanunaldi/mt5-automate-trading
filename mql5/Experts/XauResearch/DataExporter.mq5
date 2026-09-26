//+------------------------------------------------------------------+
//| DataExporter.mq5                                                 |
//| Runs inside the Strategy Tester (no "max bars in chart" limit)   |
//| and dumps closed M1 bars to Common\Files\<prefix>_<sym>_M1.csv   |
//| Places no orders.                                                |
//+------------------------------------------------------------------+
#property copyright "mt5-automate-trading"
#property version   "1.00"

input string InpPrefix = "xau_export";

int      g_file = INVALID_HANDLE;
datetime g_last = 0;

int OnInit()
  {
   string name = StringFormat("%s_%s_M1.csv", InpPrefix, _Symbol);
   g_file = FileOpen(name, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(g_file == INVALID_HANDLE)
     {
      PrintFormat("FileOpen failed %d", GetLastError());
      return(INIT_FAILED);
     }
   FileWrite(g_file, "time", "open", "high", "low", "close", "tick_volume", "spread");
   return(INIT_SUCCEEDED);
  }

void OnTick()
  {
   MqlRates r[];
   if(CopyRates(_Symbol, PERIOD_M1, 1, 1, r) != 1)
      return;
   if(r[0].time <= g_last)
      return;
   g_last = r[0].time;
   FileWrite(g_file, (long)r[0].time, r[0].open, r[0].high, r[0].low, r[0].close,
             (long)r[0].tick_volume, r[0].spread);
  }

void OnDeinit(const int reason)
  {
   if(g_file != INVALID_HANDLE)
      FileClose(g_file);
  }

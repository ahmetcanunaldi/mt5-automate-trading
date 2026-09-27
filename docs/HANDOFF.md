# Başka bilgisayarda devam etmek için devir notu (2026-09-28)

## Durum
- **Nihai algoritma: v2.2** — `docs/ALGORITHM.md` (tek doğru kaynak). XAUUSD 12 bacak + NAS100/DJ30 6 literatür
  bacağı + tampon-oranlı risk kuralı. Python: SR 1.91, CAGR %16.9, DD %7.57; MT5 tester: +$279,494, 6,630 işlem.
- **EA: `mql5/Experts/XauScalper/XauIdxPortfolio.mq5` v2.21** (canlı çalışma güvenlikleri; tester v2.20 ile birebir).
- **Sıradaki iş: demo forward test** — `deploy/forward/KURULUM.md` (MetaQuotes demo, challenge modu $100k, EA'yı
  kullanıcı ekler, haftalık `python tools/forward_report.py --shadow`).
- Tüm deneyler: `EXPERIMENTS.md` (EXP-001..103, QM-001..014, FXR-001..003). Bulgular: `docs/knowhow.md`,
  `docs/quant_findings.md`, `docs/fx_findings.md`, `docs/literature_2025_2026.md`, kurallar `docs/rules.md`.
- Dallar: `master` (hepsi), `quant-math` (master'a birleştirildi, tarihçe için duruyor).

## Kalıcı kurallar ve kullanıcı tercihleri (Claude'un yerel hafızasından)
- Hedef: FundingPips 2-Step Standard **$100k**. İç limitler: Sharpe ≥ 1.5, günlük DD %3, toplam DD %8, işlem başı
  risk ≤ %0.5, hedge yok, FP kuralları (HFT/tick scalping/grid/martingale yok), hafta sonu flat.
- Haber kuralı v2: yüksek etkili haberden 10 dk önce – 10 dk sonra yeni pozisyon yok; pozisyonlar 10 dk önce kapanır.
- Funded plan: +%3'te çekim, bakiye $100k'ya döner; **%35 tutarlılık kuralı kalıcı tasarım kısıtı**
  (`research/payout_sim.py` ile her aday değerlendirilir). Haftalık ≥ 2R hedefi (henüz karşılanmıyor, SR ≈ 4 gerekir).
- Her seçilmiş/madenlenmiş kenar **placebo testinden** geçmeli (yıl içi karıştırılmış getirilerle aynı seçim, p < 0.05);
  v2.1 bu yüzden reddedildi.
- Nihai algoritma değişince `docs/ALGORITHM.md` aynı commit'te güncellenir (sürüm + değişiklik geçmişi).
- Saf matematik araştırması: DEV 2018–2024, kilitli kutu 2025-01..2026-09 bir kez açıldı (QM-LOCKBOX).
- "Scalping" = gün içi (dakikalar–saatler), tick scalping değil.

## Güvenlik
- MCP ile bağlanan terminal (`D:\copy-trade-web-application\mt5_slave_2`) kullanıcının **GERÇEK** Vantage hesabı ve
  copy-trade uygulamasının parçası: yalnızca veri + derleme + backtest; **`trade_*` MCP araçları asla çağrılmaz**
  (`tools/mcp_client.py` bunları engeller). Forward test yalnızca ayrı demo hesapta.
- `mt5-mcp.txt` (MCP anahtarı) repoda **yok** (gitignore). Yeni bilgisayarda kendi MCP sunucunun URL/anahtarını
  aynı dosyaya yaz: `tools/mcp_client.py` okur.
- C: diski 2026-09-27'de doldu (tester "file write error [112]"). Büyük çıktılar D:'de; export aracı CSV'leri
  `data/raw_exports`'a taşır.

## Yeni bilgisayarda kurulum
1. `git clone https://github.com/ahmetcanunaldi/mt5-automate-trading.git`
2. Python 3.12: `python -m venv .venv` → `.venv\Scripts\pip install -r requirements.txt`
3. **Veri repoda değil** (`data/` gitignore; broker verisi, ~1 GB parquet + 2.5 GB ham CSV). İki yol:
   - Bu bilgisayardan `D:\mt5-automate-trading\data\` klasörünü (en azından `*.parquet`, `*.csv`, `*.pkl`,
     `fred/`; `raw_exports/` ve `ml_dataset_*` gerekmez) USB/bulut ile kopyala — en hızlısı.
   - Ya da yeniden üret: MT5 + MCP ile `tools/export_m1.py XAUUSD NAS100.r DJ30.r ...` (M1 2018+),
     `tools/fetch_history.py` (H1/D1), `tools/export_news.py` (haber takvimi); bacak önbelleği
     `python research/legcache.py --rebuild`.
4. Terminal yolları `tools/deploy.py` / `tools/export_m1.py` / `tools/forward_report.py` içinde bu bilgisayara göre
   sabit (`C:\Users\ahmet\AppData\Roaming\MetaQuotes\...`, terminal kimliği `8A8C66D5...`); yeni makinede güncelle.
5. Testler: `.venv\Scripts\python -m pytest research/tests research/quant/tests`.

## Plan dosyası
Quant-math araştırma planı: `docs/plans/quant-math-plan.md`.

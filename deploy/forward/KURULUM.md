# v2.2 demo forward test — kurulum ve izleme

EA: `XauIdxPortfolio.ex5` (v2.21 = algoritma v2.2 + canlı çalışma güvenlikleri). Mod: challenge, $100k.

## 1. Demo hesap
- MetaQuotes demo sunucusunda **$100,000**, **USD**, kaldıraç 1:100 (veya 1:30) hesap aç.
- Gerçek hesabın bağlı olduğu terminali (copy-trade slave) **kullanma**; demo için ayrı bir MT5 kurulumu kullan.

## 2. Dosyalar
| Dosya | Nereye |
|---|---|
| `XauIdxPortfolio.ex5` | demo terminal → Dosya → Veri Klasörünü Aç → `MQL5\Experts\` |
| `xau_news_server.csv`, `fomc_server.csv` | `C:\Users\ahmet\AppData\Roaming\MetaQuotes\Terminal\Common\Files\` (tüm terminaller ortak kullanır; zaten orada) |
| `fwd_challenge.set` | EA girdileri penceresinde **Yükle** |

## 3. Semboller (demo sunucusunda kontrol et)
- Piyasa Gözlemi → Semboller: altın (`XAUUSD`), Nasdaq-100 ve Dow Jones CFD adlarını bul.
- `InpNasSymbol` / `InpDjSymbol` girdilerine demo sunucusundaki adları yaz (ör. `NAS100`, `US100`, `USTEC`, `US30`).
- Endeks CFD'si yoksa `InpTradeNas=false` / `InpTradeDj=false` (o zaman yalnızca altın bacakları çalışır; beklenen
  Sharpe 1.9 yerine ≈ 1.5).

## 4. Başlatma
1. Araçlar → Seçenekler → Uzman Danışmanlar: **Algoritmik işleme izin ver**.
2. XAUUSD grafiği aç (herhangi bir zaman dilimi), EA'yı sürükle, `fwd_challenge.set` yükle, **Algo Trading** açık.
3. Uzmanlar sekmesinde şunları gör:
   - `[Init] server offset GMT+3.0 h, expected GMT+3.0 h` — farklıysa EA **işlem açmaz** (oturum saatleri, haber
     penceresi ve günlük barlar araştırmadaki New York kapanışı saatine göre). Bu durumda bana mesajı gönder.
   - `[News] ... events loaded, last=2026.12.23` ve `[FOMC] ... decisions loaded`.
   - her sembol için `tick value / vol min / step / filling` satırı.
4. Terminal açık kalmalı (PC uykuya geçmemeli) veya bir VPS kullanılmalı.

## 5. Yeniden başlatma / güvenlik
- EA açık pozisyonları, günlük sayaçları ve **toplam stop** durumunu `Common\Files\fwd_state.csv` dosyasında tutar;
  terminal kapanıp açılınca kaldığı yerden devam eder.
- Yeni bir challenge (sıfırdan) başlatırken `fwd_state.csv`, `fwd_entries.csv`, `fwd_deals.csv` dosyalarını sil.
- Haber dosyası 2026-12-23'e kadar, FOMC dosyası 2026-12-09 toplantısına kadar kapsıyor; Aralık başında
  güncellenecek (EA 3 hafta kala uyarı verir, dosya bittiğinde işlem açmaz).

## 6. İzleme (haftalık)
- EA her girişte `fwd_entries.csv`'ye yazar, her yeni gün `fwd_deals.csv`'yi yeniler (Common\Files).
- Rapor: `python tools/forward_report.py --shadow` → `reports/forward/<tarih>/`:
  challenge durumu (bakiye, Faz-1 ilerlemesi, statik DD, en kötü gün, risk ≤ %0.5, hedge yok, hafta sonu yok),
  bacak bazında sonuçlar ve **gölge karşılaştırma**: aynı EA, MT5 tester'da Vantage verisiyle aynı tarihlerde koşar;
  demonun aynı sinyalleri (sembol, bacak, yön, ±3 dk) alıp almadığı ölçülür.

## 7. Başarı / durdurma ölçütleri
- En az **8–12 hafta** (backtest ortalaması haftada ≈ 17 işlem → ≈ 130–200 işlem).
- Sinyal eşleşmesi ≥ %90 (farklar: broker verisi, spread, haber saatleri).
- Kural ihlali 0 (risk > %0.5, hedge, hafta sonu pozisyonu, haber penceresinde giriş).
- İşlem başına R ve kayma, backtest dağılımıyla uyumlu (ortalama ≈ +0.04 R/işlem, gürültü büyük: 200 işlemde
  standart hata ≈ ±0.03 R; 12 haftalık getiri tek başına algoritmayı doğrulamaya yetmez — asıl ölçüt sinyal eşleşmesi
  ve icra kalitesi).
- Durdur: statik DD %5'i aşarsa veya tekrarlayan emir reddi / kural ihlali olursa → incele.

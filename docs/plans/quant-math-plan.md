# Yan dal "quant-math": saf fiyat/hacim zaman serisinden matematiksel ve istatistiksel çıkarım

## Context
Ana algoritma v2 (XAUUSD best13 + NAS100/DJ30 bacakları, SR 1.91) büyük ölçüde gözlemsel düzenliliklere dayanıyor
(takvim, "o gün yükselir"). EXP-099: Sharpe'ının ~%60'ı yükselen piyasada long durmaktan geliyor. Kullanıcı artık
**yalnızca fiyat, hacim ve geçmişten** matematiksel çıkarım istiyor: stokastik diferansiyel denklemler, filtreleme,
optimal portföy/kontrol, istatistiğin sınırlarını zorlayan doğrulama. Bu iş **ayrı bir git dalında** yapılacak; ana dal
(v2, EA) dokunulmadan kalır. Amaç: v2'den bağımsız, modele dayalı sinyaller bulmak; bulunamazsa bunu da kanıtla göstermek.

Beklenti (dürüstçe): yön öngörüsü zayıftır (ML AUC 0.51); matematiğin güvenilir kazandığı yerler **volatilite tahmini**,
**ortalamaya dönen (eşbütünleşik) spread'ler**, **rejim filtreleme** ve **kısıtlı optimal pozisyon boyutlama**dır.
Plan bu sıraya göre önceliklendirildi.

## Değişmeyen kurallar
- İşlem başı risk ≤ %0.5, günlük %3 / toplam %8 iç limit (FP: %5/%10), sembol başına hedge yok, haber v2 (±10 dk giriş
  yok, 10 dk önce kapat), hafta sonu flat, MCP terminali yalnız veri + backtest (`tools/mcp_client.py` trade_* bloklu).
- Tüm backtestler mevcut icra motorlarıyla: `research/engine.py`, `research/engine_multi.py` (maliyet, spread,
  swap, FP korumaları, çekim/tutarlılık simülasyonu `research/payout_sim.py`).
- Her deney `EXPERIMENTS.md`'ye "QM-###" olarak; damıtılmış bulgular `docs/quant_findings.md`'ye.

## Evren ve veri
- Semboller: XAUUSD, XAGUSD, NAS100, DJ30, GER40, EURUSD, USDJPY (M1 2018–2026 mevcut, `research/symbols.py`)
  + SP500.r M1 tester ile dışa aktarılacak (`tools/export_m1.py`, endeks eşbütünleşmesi için).
- Hacim: M1 tick_volume (tüm dönem); tick dönemi (2024-12→) için yukarı/aşağı tick sayıları, spread
  (`data/*_M1T.parquet`, `research/tick_features.py`).
- **Kilitli kutu (lockbox): 2025-01-01 → 2026-09-25** bu dalın model geliştirmesinde hiç kullanılmaz; yalnızca
  final adaylar için bir kez açılır. Geliştirme: 2018–2024 (walk-forward içinde eğitim/test).

## Mimari (yeni dal `quant-math`)
```
research/quant/
  data.py        çok sembollü hizalı getiriler (M1→M5/M15/H1/D1), realized variance, bipower variation, hacim
  validate.py    doğrulama çekirdeği (aşağıda) + deneme defteri (her konfigürasyon sayılır)
  vol.py         HAR-RV, GARCH/EGARCH/GJR (arch), rough-vol Hurst tahmini, vol tahmin değerlendirmesi (QLIKE)
  filters.py     Kalman local-level/local-trend drift, Markov-switching (Hamilton), HMM (hmmlearn), Wonham filtresi
  ou.py          OU MLE, yarı ömür, Bertram optimal giriş/çıkış eşikleri, maliyetli bant (HJB çözümü)
  coint.py       Engle-Granger/Johansen, dinamik hedge oranı (Kalman), spread üretimi
  jumps.py       Lee–Mykland sıçrama testi, Hawkes süreci (MLE) ile yoğunluk, sıçrama sonrası davranış
  dependence.py  varyans oranı (Lo–MacKinlay, çoklu ufuk), otokorelasyon spektrumu, DFA/Hurst, karşılıklı bilgi,
                 transfer entropi, Hayashi–Yoshida lead-lag
  portfolio.py   Ledoit-Wolf ortalama-varyans, risk parity, HRP, drawdown-kısıtlı Kelly (Grossman–Zhou), CPPI
  signals.py     modellerden sinyal → mevcut motorun sinyal formatı (dir, sl, tp, hold_min, trail, risk_mult, sym)
research/quant/tests/   her modül için sentetik veri testleri (bilinen parametreli OU/GARCH/HMM'yi geri kazanma)
```
Kurulum: `.venv` içine `arch`, `hmmlearn` (gerekirse `PyWavelets`) — küçük, saf Python/C paketleri.

## Doğrulama çekirdeği (istatistiğin sınırı) — `validate.py`
Her aday şunlardan geçer; hepsi raporlanır:
1. **Walk-forward** (genişleyen pencere, yıllık yeniden tahmin) + **purge/embargo** (etiket ufku kadar).
2. **Combinatorial Purged CV** → OOS Sharpe dağılımı ve **PBO** (backtest overfitting olasılığı, Bailey et al.); hedef PBO < 0.2.
3. **Deflated Sharpe Ratio** (deneme sayısı, çarpıklık, basıklık düzeltmeli); hedef DSR > 0.95.
4. **Placebo / null modeller**: (a) yıl içi karıştırma, (b) blok bootstrap, (c) **tahmin edilmiş null süreçten
   sentetik veri** (GARCH veya HAR ile simüle edilmiş getiriler: sinyal volatilite yapısından mı geliyor?). p < 0.05.
5. **Hansen SPA / White Reality Check** aynı ailedeki tüm konfigürasyonlar için.
6. **Drift kıyası**: aynı maruziyetle "her zaman long" benchmark'ını (EXP-099 yöntemi) anlamlı geçmeli.
7. Maliyet ×1.5 stresi, lockbox tek seferlik açılış.

## Fazlar (sırası = beklenen değer)
**Faz 0 — Altyapı:** dal, `data.py`, `validate.py`, deneme defteri, sentetik testler; SP500 dışa aktarma.

**Faz 1 — Öngörülebilirlik haritası (işlem yok, sadece çıkarım):** her sembol × ufuk (1 dk–5 gün) × seans ×
volatilite rejimi için varyans oranı, otokorelasyon, Hurst/DFA, karşılıklı bilgi ve transfer entropi (varlıklar
arası, gecikmeli). Çıktı: "bağımlılık nerede var" ısı haritası + sonraki fazlara hangi ufukların taşınacağı.

**Faz 2 — Volatilite (en öngörülebilir bileşen):** HAR-RV / GARCH / rough-vol tahminleri, QLIKE ile karşılaştırma;
(a) **volatiliteyle ölçeklenmiş maruziyet** (Moreira–Muir) hem basit drift pozisyonlarına hem v2 bacaklarına,
(b) vol tahmini ile dinamik stop/hedef genişliği, (c) varyans riski primi yerine gerçekleşen vol sürprizlerinin
sonraki getiriyle ilişkisi (leverage/feedback etkisi).

**Faz 3 — Drift filtreleme ve rejimler (SDE: dS = μ_t S dt + σ_t S dW):** Kalman ile gizli μ_t tahmini, Markov-switching
μ/σ (Hamilton), HMM ve sürekli zamanda **Wonham filtresi** ile optimal rejim olasılığı → pozisyon = f(P(boğa), σ̂).
Zaman serisi momentumunun optimal filtre versiyonu (sinyal = μ̂_t / σ̂_t²). Walk-forward, drift kıyasına karşı.

**Faz 4 — Ortalamaya dönüş ve istatistiksel arbitraj (OU):** NAS100–DJ30–SP500–GER40 ve XAU–XAG eşbütünleşme,
Kalman dinamik hedge oranı; spread için OU MLE, yarı ömür, **Bertram (2010) optimal eşikleri** ve maliyetli
HJB bantları. İki bacaklı işlemler farklı sembollerde olduğundan FP "hedge yok" kuralına uyar (sembol başına tek yön);
her bacak kendi SL'iyle ≤ %0.5 risk. Gün içi ölçekte: VWAP'tan sapmanın OU dinamiği.

**Faz 5 — Sıçramalar, Hawkes ve hacim:** Lee–Mykland ile sıçrama tespiti, sıçrama sonrası devam/dönüş; Hawkes
yoğunluğuyla (tick_volume / tick varışları) volatilite patlaması tahmini; hacim koşullu otokorelasyon (Campbell–
Grossman–Wang), tick döneminde emir akışı dengesizliği ve Kyle lambda'sı ile kısa vadeli getiri.

**Faz 6 — Optimal portföy ve kontrol:** Faz 2–5'ten geçen sinyaller + v2 bacakları "varlık" olarak; Ledoit-Wolf MV,
risk parity, HRP karşılaştırması (walk-forward kovaryans). FP kısıtlarına özel: **drawdown-kısıtlı Kelly
(Grossman–Zhou 1993)** — maruziyet ∝ (servet − %92 taban), günlük %3 kısıtı için CPPI benzeri gün içi taban;
çekim + %35 tutarlılık altında dinamik programlama ile risk politikası. İşlem başı %0.5 üst sınır korunur.

**Faz 7 — Karar:** Tüm kapıları (Sharpe ≥ 1.5 veya v2'ye anlamlı katkı, DD, PBO, DSR, placebo, drift kıyası) geçen
modeller lockbox'ta bir kez test edilir; geçenler ana dala önerilir ve `docs/ALGORITHM.md` ancak o zaman güncellenir.
Geçmeyenler de raporlanır (neden geçmediği bilgi değeri taşır).

## Kritik dosyalar
- Yeni: `research/quant/*.py`, `research/quant/tests/*`, `docs/quant_findings.md`
- Yeniden kullanılacak: `research/engine.py::run/prepare_exec`, `research/engine_multi.py::align/run`,
  `research/symbols.py::load_m1/prepare/COSTS`, `research/metrics.py` (summarize, challenge_sim, weekly_stats),
  `research/payout_sim.py::summarize_payouts`, `research/plots.py::equity_report`, `research/lab.py::save_experiment`,
  `research/features.py::atr/ema`, `tools/export_m1.py`
- Değişmeyecek (ana dal): `docs/ALGORITHM.md`, `mql5/Experts/XauScalper/XauIdxPortfolio.mq5`

## Doğrulama
- `pytest research/tests research/quant/tests`: her model sentetik veride bilinen parametreleri geri kazanmalı
  (OU θ/μ/σ, GARCH α/β, HMM geçiş matrisi, Hawkes α/β, Bertram eşikleri analitik değerlerle).
- Her QM deneyi: metrics.json + equity grafiği (günlük/total DD tabanlarıyla) + walk-forward tablosu + PBO/DSR/placebo p.
- Ana dalın testleri ve v2 sonuçları (SR 1.914) dal değişikliklerinden etkilenmemeli (regresyon kontrolü).
- Faz sonlarında kullanıcıya kısa rapor; ana dala birleştirme yalnızca kullanıcı onayıyla.
